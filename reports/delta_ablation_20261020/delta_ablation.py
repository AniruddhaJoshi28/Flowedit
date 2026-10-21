#!/usr/bin/env python
"""
2 × 2 factorial ablation: respelling × δ injection, on the genuine F5-TTS backbone.

  A  original spelling,  no δ        (F5-TTS as is)
  B  respelled word,     no δ
  C  original spelling,  δ injected
  D  respelled word,     δ injected   (current FlowEdit pipeline)

Retrieval is bypassed: for every sentence the stored correction of the *correct* pronunciation is
supplied directly (its spelling and its F5-space word δ*), so retrieval mistakes cannot enter.
δ is placed on the word's character rows exactly as HopfieldRefiner does for character-level
backbones (same alignment, correction scale, element clamp and L2 cap). Same seed, steps, CFG,
speed and speaker per sentence across all four conditions.

Injection check: every synthesis counts the backbone's span-injection events and the injected
span lengths; A and B must show none, C and D must inject exactly the word's characters.

Measures
  pronunciation   target vowel from a phoneme CTC model inside the Whisper word window
                  (unscorable outputs reported separately)
  preservation    non-target words aligned to condition A by Whisper word timings; MFCC-DTW
                  distance per aligned word, and share of the sentence's non-target words recovered
  speaker         cosine similarity of WeSpeaker embeddings to the speaker reference
  transfer        held-out sentences (never used to optimise δ) vs the correction sentences

Outputs: ablation_rows.csv, ablation_summary.json (paired tables, McNemar exact tests).
"""

import argparse
import csv
import difflib
import json
import logging
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flowedit.config import FlowEditConfig  # noqa: E402
from flowedit.pipeline.inference import FlowEditInference  # noqa: E402
import sense_context_eval as sce  # noqa: E402

CONDITIONS = {"A": (False, False), "B": (True, False), "C": (False, True), "D": (True, True)}


class InjectionCounter(logging.Handler):
    """Counts F5TTS span-injection log events (one per text-embedding forward with δ applied)."""

    def __init__(self):
        super().__init__()
        self.events, self.lengths = 0, set()

    def emit(self, record):
        msg = record.getMessage()
        if "[F5TTS Span Injection]" in msg:
            self.events += 1
            for n in re.findall(r"\(\d+, \d+, (\d+)\)", msg):
                self.lengths.add(int(n))

    def reset(self):
        self.events, self.lengths = 0, set()


def respell(text, word, spelling):
    def case(m):
        o = m.group()
        return spelling.capitalize() if o[0].isupper() else spelling
    return re.sub(rf"\b{word}\b", case, text, count=1, flags=re.IGNORECASE)


def build_delta(mem, backbone, entry, synth_text, word_surface):
    """Character-level δ for synth_text, as HopfieldRefiner builds it for F5-TTS."""
    m = re.search(rf"\b{re.escape(word_surface)}\b", synth_text, flags=re.IGNORECASE)
    s, e = m.start(), m.end()
    wd = entry.word_delta.float()
    src = wd[0] if wd.dim() == 3 else wd
    scale = float(mem.config.correction_scale)
    delta = torch.zeros(1, len(synth_text), backbone.embedding_dim)
    delta[0, s:e, :] = mem._align_delta(src, e - s) * scale
    clamp_val, norm_cap = backbone.delta_bounds
    delta = torch.clamp(delta, -clamp_val, clamp_val)
    active = delta.abs().sum(-1) > 1e-5
    n = delta[active].norm().item()
    if n > norm_cap:
        delta = delta * (norm_cap / n)
    return delta, e - s


def mfcc(y):
    import librosa
    return librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13)


def dtw_dist(a, b):
    import librosa
    if a.shape[1] < 2 or b.shape[1] < 2:
        return None
    D, wp = librosa.sequence.dtw(X=a, Y=b, metric="euclidean")
    return float(D[-1, -1] / len(wp))


def norm_word(w):
    return re.sub(r"[^a-z']", "", w.lower())


def mcnemar(pairs):
    """Exact two-sided McNemar test on paired booleans [(x_correct, y_correct)]."""
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = b + c
    if n == 0:
        return {"x_only": b, "y_only": c, "p": 1.0}
    k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"x_only": b, "y_only": c, "p": round(p, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", required=True, help="evaluation memory with real F5 corrections (read only)")
    ap.add_argument("--speakers", nargs="+", required=True, help="speaker reference WAVs; the first is the correction speaker")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "wavs"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(args.out, "run.log")), logging.StreamHandler()])
    torch.backends.cudnn.deterministic = True
    sce.isolate_runtime()

    cfg = FlowEditConfig()
    cfg.backbone.backbone_type = "f5tts"
    inf = FlowEditInference(cfg)
    inf.load()
    mem, bb = inf.memory, inf.backbone
    mem.load(args.memory)
    ident = sce.checkpoint_identity(bb)
    counter = InjectionCounter()
    logging.getLogger("flowedit.backbone.f5tts_wrapper").addHandler(counter)

    # Correct correction per (word, vowel): the entry with that spelling and an F5-space δ*
    corr = {}
    for (word, vowel), spelling in sce.SPELL.items():
        cands = [e for e in mem.entries if e.word.lower() == word and (e.phonetic_text or "").lower() == spelling
                 and e.delta_space == mem.space and e.word_delta is not None]
        if cands:
            corr[(word, vowel)] = max(cands, key=lambda e: e.access_count)
    logging.info(f"corrections with F5 δ*: {[(k, v.carrier_texts) for k, v in corr.items()]}")

    items = []   # (sid, kind, word, vowel, text)
    for sid, cat, text, expected in sce.TESTS:
        if len(expected) == 1 and tuple(expected[0]) in corr:
            items.append((sid, "held_out", expected[0][0], expected[0][1], text))
    optimised = {c for _, _, c, _ in sce.CORRECTIONS}   # sentences the F5 correction loop optimised δ* on
    for (word, vowel), e in corr.items():
        for i, c in enumerate(c for c in (e.carrier_texts or []) if c in optimised):
            items.append((f"S_{word}_{vowel}_{i}", "correction_sentence", word, vowel, c))

    import whisper
    from pyannote.audio import Model, Inference
    asr = whisper.load_model(os.environ.get("EVAL_WHISPER", "medium"))
    judge = sce.PhonemeJudge(asr, "cuda" if torch.cuda.is_available() else "cpu")
    spk_model = Inference(Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM"), window="whole")
    spk_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def spk_emb(y, sr):
        import librosa
        y = librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=sr, target_sr=16000)
        w = torch.from_numpy(y)[None]
        e = np.asarray(spk_model({"waveform": w, "sample_rate": 16000})).reshape(-1)
        return e / (np.linalg.norm(e) + 1e-9)

    import librosa
    rows = []
    for spk_i, spk in enumerate(args.speakers):
        spk_name = os.path.splitext(os.path.basename(spk))[0]
        cond = bb.get_speaker_embedding(spk, language="en")
        ref_y, ref_sr = sf.read(cond.get("processed_audio_path") or spk, dtype="float32", always_2d=True)
        ref_emb = spk_emb(ref_y.mean(1), ref_sr)
        for sid, kind, word, vowel, text in items:
            entry = corr[(word, vowel)]
            spelling = sce.SPELL[(word, vowel)]
            out = {}
            for cname, (do_respell, do_delta) in CONDITIONS.items():
                synth_text = respell(text, word, spelling) if do_respell else text
                surface = spelling if do_respell else word
                delta, span_len = build_delta(mem, bb, entry, synth_text, surface) if do_delta else (None, 0)
                counter.reset()
                torch.manual_seed(sce.SEED)
                wav, sr = bb.synthesize_direct(text=synth_text, speaker_conditioning=cond, language="en",
                                               text_embedding_delta=delta, **sce.PARAMS)
                y = wav.squeeze().cpu().numpy()
                path = os.path.join(args.out, "wavs", f"{spk_name}_{sid}_{cname}.wav")
                sf.write(path, y, sr)
                inj_ok = (counter.events == 0) if not do_delta else (counter.events > 0 and counter.lengths == {span_len})
                y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
                tr = asr.transcribe(path, word_timestamps=True, language="en")
                words = [w for s in tr["segments"] for w in s["words"]]
                verdict = judge.judge(path, [(word, vowel)])["occ"][0]
                out[cname] = {"y16": y16, "words": words, "path": path}
                rows.append({
                    "speaker": spk_name, "correction_speaker": spk_i == 0, "id": sid, "kind": kind, "word": word,
                    "expected": vowel, "condition": cname, "synth_text": synth_text,
                    "injection_events": counter.events, "injection_ok": inj_ok,
                    "judged": verdict["verdict"], "vowel": verdict.get("vowel"), "window": verdict.get("window"),
                    "correct": "" if verdict["verdict"] == "MANUAL" else int(verdict["verdict"] == vowel),
                    "speaker_sim": float(spk_emb(y, sr) @ ref_emb), "whisper": tr["text"].strip(),
                    "wav": os.path.relpath(path, args.out),
                })
            # Non-target preservation against condition A (aligned words, target excluded)
            target_re = sce.WORD_RE[word]
            src_words = [norm_word(w) for w in text.split() if not target_re.match(norm_word(w))]
            base = [(norm_word(w["word"]), w) for w in out["A"]["words"] if not target_re.match(norm_word(w["word"]))]
            for r in rows[-4:]:
                o = out[r["condition"]]
                cur = [(norm_word(w["word"]), w) for w in o["words"] if not target_re.match(norm_word(w["word"]))]
                recovered = sum(b.size for b in difflib.SequenceMatcher(
                    a=src_words, b=[c[0] for c in cur], autojunk=False).get_matching_blocks())
                r["nontarget_recovered"] = recovered / max(1, len(src_words))
                if r["condition"] == "A":
                    r["nontarget_dtw_vs_A"] = 0.0
                    continue
                sm = difflib.SequenceMatcher(a=[b[0] for b in base], b=[c[0] for c in cur], autojunk=False)
                dists = []
                for blk in sm.get_matching_blocks():
                    for k in range(blk.size):
                        wa, wb = base[blk.a + k][1], cur[blk.b + k][1]
                        ya = out["A"]["y16"][int(wa["start"] * 16000):int(wa["end"] * 16000)]
                        yb = o["y16"][int(wb["start"] * 16000):int(wb["end"] * 16000)]
                        d = dtw_dist(mfcc(ya), mfcc(yb)) if len(ya) > 400 and len(yb) > 400 else None
                        if d is not None:
                            dists.append(d)
                r["nontarget_dtw_vs_A"] = float(np.mean(dists)) if dists else None
                r["nontarget_aligned_words"] = len(dists)
            logging.info(f"[{spk_name} {sid}] " + " ".join(f"{r['condition']}:{r['judged']}" for r in rows[-4:]))

    with open(os.path.join(args.out, "ablation_rows.csv"), "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # ── Summary ──
    idx = {(r["speaker"], r["id"], r["condition"]): r for r in rows}
    groups = defaultdict(list)
    for r in rows:
        if r["condition"] == "A":
            groups[(r["speaker"], r["kind"])].append(r["id"])

    def table(sel_keys, filt=lambda r: True):
        res = {}
        for c in CONDITIONS:
            rs = [idx[(s, i, c)] for s, i in sel_keys if filt(idx[(s, i, c)])]
            sc = [r for r in rs if r["correct"] != ""]
            d = [r["nontarget_dtw_vs_A"] for r in rs if r.get("nontarget_dtw_vs_A") is not None]
            res[c] = {"n": len(rs), "correct": sum(r["correct"] for r in sc), "wrong": len(sc) - sum(r["correct"] for r in sc),
                      "unscorable": len(rs) - len(sc),
                      "acc_scorable": round(sum(r["correct"] for r in sc) / len(sc), 3) if sc else None,
                      "speaker_sim": round(float(np.mean([r["speaker_sim"] for r in rs])), 4) if rs else None,
                      "nontarget_dtw_vs_A": round(float(np.mean(d)), 3) if d else None,
                      "nontarget_recovered": round(float(np.mean([r["nontarget_recovered"] for r in rs])), 3) if rs else None}
        return res

    def paired(sel_keys, x, y):
        pairs = []
        for s, i in sel_keys:
            rx, ry = idx[(s, i, x)], idx[(s, i, y)]
            if rx["correct"] != "" and ry["correct"] != "":
                pairs.append((bool(rx["correct"]), bool(ry["correct"])))
        sims = [(idx[(s, i, x)]["speaker_sim"], idx[(s, i, y)]["speaker_sim"]) for s, i in sel_keys]
        dtw = [(idx[(s, i, x)].get("nontarget_dtw_vs_A"), idx[(s, i, y)].get("nontarget_dtw_vs_A")) for s, i in sel_keys]
        dtw = [(a, b) for a, b in dtw if a is not None and b is not None]
        return {"pronunciation_mcnemar": {**mcnemar(pairs), "n_paired_scorable": len(pairs)},
                "speaker_sim_mean_diff": round(float(np.mean([b - a for a, b in sims])), 4) if sims else None,
                "nontarget_dtw_mean_diff": round(float(np.mean([b - a for a, b in dtw])), 3) if dtw else None}

    summary = {"checkpoint": ident, "injection_all_ok": all(r["injection_ok"] for r in rows),
               "corrections": {f"{w}|{v}": e.carrier_texts for (w, v), e in corr.items()}, "results": {}}
    for (spk, kind), ids in sorted(groups.items()):
        keys = [(spk, i) for i in ids]
        blk = {"conditions": table(keys), "B_vs_D": paired(keys, "B", "D"), "A_vs_C": paired(keys, "A", "C"),
               "by_word_pron": {}}
        for (word, vowel) in corr:
            sub = [(s, i) for s, i in keys if idx[(s, i, "A")]["word"] == word and idx[(s, i, "A")]["expected"] == vowel]
            if sub:
                blk["by_word_pron"][f"{word}|{vowel}"] = {"conditions": table(sub), "B_vs_D": paired(sub, "B", "D"),
                                                          "A_vs_C": paired(sub, "A", "C")}
        summary["results"][f"{spk}|{kind}"] = blk
    json.dump(summary, open(os.path.join(args.out, "ablation_summary.json"), "w"), indent=2, default=str)
    logging.info("DONE")


if __name__ == "__main__":
    main()
