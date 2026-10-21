#!/usr/bin/env python
"""
Real F5-TTS evaluation of Context-Robust Hopfield Retrieval (baseline / shadow / active).

Phases (all on the genuine F5-TTS checkpoint, one model instance for every mode):
  0. Checkpoint identity: path, sha256, architecture, parameter count, embedding shapes, device.
  1. Real correction memory: crops human reference clips (one per pronunciation of the homograph) and
     runs the unmodified CorrectionLoop on F5-TTS. Stored vectors come only from that optimisation.
  2. Synthesis: every sentence in raw F5 (no memory), baseline, shadow and active retrieval modes with
     identical checkpoint, speaker reference, inference parameters and seed.
  3. Evaluation: target-word vowel from a phoneme CTC recogniser (no text prior) inside Whisper word
     windows; ambiguous cases are marked MANUAL. Invariants: stored vectors unchanged, baseline == shadow,
     gate values identical across modes, unrelated attention mass unchanged, single-sense words on the
     baseline path.

The production memory and spelling dictionary are never written; the external semantic models
(sense encoder, sense lexicon) and the rule-based spelling store are disabled for the run.
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
import time

os.environ["FLOWEDIT_SENSE_ENCODER"] = ""
os.environ["FLOWEDIT_SENSE_LEXICON"] = ""
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flowedit.config import FlowEditConfig  # noqa: E402
from flowedit.memory.s3_storage import s3_spelling_store  # noqa: E402
from flowedit.pipeline.correction_loop import CorrectionLoop  # noqa: E402
from flowedit.pipeline.inference import FlowEditInference  # noqa: E402

logger = logging.getLogger("ctx_f5_eval")

MODES = ("baseline", "shadow", "active")
SEED = 42

# (id, category, sentence, expected pronunciation of each "lead" occurrence in order: LEED=/liːd/, LED=/lɛd/)
SENTENCES = [
    ("A1", "same_pron_diff_neighbors", "Maria is the lead of the design team.", ["LEED"]),
    ("A2", "same_pron_diff_neighbors", "He was chosen to lead the new research project.", ["LEED"]),
    ("A3", "same_pron_diff_neighbors", "Our lead engineer approved the final plan.", ["LEED"]),
    ("A4", "same_pron_diff_neighbors", "The old water pipes in this house are made of lead.", ["LED"]),
    ("A5", "same_pron_diff_neighbors", "Paint that contains lead is dangerous for children.", ["LED"]),
    ("A6", "same_pron_diff_neighbors", "The fishing weights were cast from solid lead.", ["LED"]),
    ("B1", "diff_pron_same_spelling", "The toy is made of lead.", ["LED"]),
    ("B2", "diff_pron_same_spelling", "She is the lead of data science.", ["LEED"]),
    ("B3", "diff_pron_same_spelling", "Lead poisoning is a serious health risk.", ["LED"]),
    ("B4", "diff_pron_same_spelling", "Lead the way, and we will follow you.", ["LEED"]),
    ("C1", "long_distance", "After years of careful study and many failed attempts in the laboratory, "
                            "the chemists finally confirmed that the heavy grey metal was lead.", ["LED"]),
    ("C2", "long_distance", "After years of training junior staff and running every weekly meeting "
                            "for the department, Priya was finally promoted to team lead.", ["LEED"]),
    ("C3", "long_distance", "The engineers who tested the soil samples from the old battery factory "
                            "found dangerous traces of lead.", ["LED"]),
    ("D1", "multi_homograph", "The lead researcher found lead in the old pipes.", ["LEED", "LED"]),
    ("D2", "multi_homograph", "Our team lead says the water contains lead.", ["LEED", "LED"]),
    ("D3", "multi_homograph", "This toy is made of lead, and she is the lead of data science.", ["LED", "LEED"]),
    ("E1", "control_single_sense", "Bispecific antibodies are becoming an important approach in cancer treatment.", []),
    ("E2", "control_single_sense", "Rybervant is approved for the treatment of lung cancer.", []),
    ("E3", "control_no_memory_word", "The weather is pleasant today.", []),
]

LEED_V = {"iː", "i", "ɪː", "iə"}
LED_V = {"ɛ", "e", "ɛː", "æ", "ɛə"}
LEAD_LIKE = re.compile(r"^(lead|led|leed|lid|lied|lede|lady|leads|lead's)$")


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tensor_digest(t):
    if t is None:
        return None
    return hashlib.sha256(t.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()[:16]


def memory_digest(mem):
    return [
        {"idx": i, "word": e.word, "phonetic": e.phonetic_text, "key": tensor_digest(e.key),
         "value": tensor_digest(e.value), "word_delta": tensor_digest(e.word_delta),
         "full_delta": tensor_digest(e.full_delta)}
        for i, e in enumerate(mem.entries)
    ]


def soundfile_torchaudio_load():
    """torchaudio>=2.9 decodes only through torchcodec/FFmpeg, which this host cannot load; decode
    WAV with soundfile instead (same (channels, frames) float32 tensor and sample rate)."""
    import torchaudio

    def _load(path, *args, **kwargs):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sr

    torchaudio.load = _load


def disable_spelling_store():
    """Rule-based respelling store would confound retrieval; keep it empty and never persist it."""
    s3_spelling_store._dictionary = {}
    s3_spelling_store._ensure_loaded = lambda: None
    s3_spelling_store._save_local_cache = lambda: False


# ─── Phase 0 ────────────────────────────────────────────────────────────────

def checkpoint_identity(backbone):
    api = backbone.tts_api
    if api is None:
        raise RuntimeError("F5-TTS API not loaded (test stub fallback) - refusing to evaluate")
    ema = getattr(api, "ema_model", None)
    tr = getattr(ema, "transformer", None)
    if tr is None or "xtts" in type(backbone).__name__.lower():
        raise RuntimeError(f"Not an F5-TTS backbone: {type(backbone).__name__}")
    ckpt = getattr(api, "ckpt_file", None) or backbone._resolve_f5_paths()[0]
    emb = backbone.encode_text("She is the lead of data science.", "en")
    te = tr.text_embed.text_embed
    info = {
        "backbone_class": type(backbone).__name__,
        "api_class": f"{type(api).__module__}.{type(api).__name__}",
        "cfm_class": type(ema).__name__,
        "transformer_class": f"{type(tr).__module__}.{type(tr).__name__}",
        "checkpoint": os.path.abspath(ckpt),
        "checkpoint_bytes": os.path.getsize(ckpt),
        "checkpoint_sha256": sha256(ckpt),
        "params_total": sum(p.numel() for p in ema.parameters()),
        "transformer_dim": getattr(tr, "dim", None),
        "transformer_depth": getattr(tr, "depth", None) or len(getattr(tr, "transformer_blocks", [])),
        "text_embed_weight": list(te.weight.shape),
        "encode_text_shape": list(emb.shape),
        "embedding_dim": backbone.embedding_dim,
        "embedding_space": backbone.embedding_space,
        "character_level": backbone.is_character_level,
        "device": str(next(ema.parameters()).device),
        "dtype": str(next(ema.parameters()).dtype),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
    }
    if info["params_total"] < 100e6:
        raise RuntimeError(f"Parameter count {info['params_total']} too small for F5-TTS")
    return info


# ─── Phase 1 ────────────────────────────────────────────────────────────────

def crop_reference_clips(aligner_model, src, out_dir):
    """Crop one clip per pronunciation of 'lead' from a human recording, by Whisper word times."""
    import librosa
    y, sr = librosa.load(src, sr=24000)
    words = [w for seg in aligner_model.transcribe(src, word_timestamps=True)["segments"] for w in seg["words"]]
    toks = [re.sub(r"[^a-z']", "", w["word"].lower()) for w in words]
    leads = [i for i, t in enumerate(toks) if t == "lead"]
    if len(leads) < 2:
        raise RuntimeError(f"Expected two 'lead' occurrences in {src}, Whisper words: {toks}")
    clips = []
    # Clip 1: "this toy is made up of lead"; clip 2: "is the lead of data science"
    first, second = leads[0], leads[1]
    start1 = max(i for i in range(first) if toks[i] == "this")
    start2 = second - 2 if toks[second - 2] == "is" else second - 1
    end2 = next((i for i in range(second, len(toks)) if toks[i] == "science"), min(second + 4, len(toks) - 1))
    for name, a, b in (("ref_lead_metal.wav", start1, first), ("ref_lead_leader.wav", start2, end2)):
        t0, t1 = max(0.0, words[a]["start"] - 0.10), words[b]["end"] + 0.15
        path = os.path.join(out_dir, name)
        sf.write(path, y[int(t0 * sr):int(t1 * sr)], sr)
        clips.append({"path": path, "t0": round(t0, 3), "t1": round(t1, 3),
                      "transcript": " ".join(toks[a:b + 1])})
    return clips


# ─── Phase 3 helpers ────────────────────────────────────────────────────────

class PhonemeJudge:
    """Phoneme CTC (wav2vec2 espeak) with frame timing + Whisper word windows."""

    def __init__(self, whisper_model, device):
        from transformers import AutoModelForCTC, AutoProcessor
        name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForCTC.from_pretrained(name).to(device).eval()
        self.device = device
        self.whisper = whisper_model
        self.model_name = name

    def phones_with_times(self, y16):
        with torch.no_grad():
            inp = self.proc(y16, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
            logits = self.model(inp).logits[0]
        ids = logits.argmax(-1).cpu().tolist()
        step = len(y16) / 16000 / max(1, len(ids))
        pad = self.proc.tokenizer.pad_token_id
        out, prev = [], None
        for k, i in enumerate(ids):
            if i != prev and i != pad:
                out.append((self.proc.tokenizer.convert_ids_to_tokens(i), k * step))
            prev = i
        return out

    def judge(self, wav_path, n_expected):
        import librosa
        y16, _ = librosa.load(wav_path, sr=16000)
        phones = self.phones_with_times(y16)
        res = self.whisper.transcribe(wav_path, word_timestamps=True, language="en")
        words = [w for seg in res["segments"] for w in seg["words"]]
        cands = [w for w in words if LEAD_LIKE.match(re.sub(r"[^a-z']", "", w["word"].lower()))]
        verdicts = []
        for k in range(n_expected):
            if len(cands) != n_expected:
                verdicts.append({"verdict": "MANUAL", "reason": f"whisper found {len(cands)} lead-like words",
                                 "window_phones": ""})
                continue
            w = cands[k]
            win = [p for p, t in phones if w["start"] - 0.10 <= t <= w["end"] + 0.10]
            vowel = None
            for j, p in enumerate(win[:-1]):
                if p == "l" and win[j + 1] not in ("l",):
                    vowel = win[j + 1]
                    break
            if vowel in LEED_V:
                v = "LEED"
            elif vowel in LED_V:
                v = "LED"
            else:
                v = "MANUAL"
            verdicts.append({"verdict": v, "vowel": vowel, "window_phones": " ".join(win),
                             "whisper_word": w["word"].strip(), "t": [round(w["start"], 2), round(w["end"], 2)],
                             "reason": "" if v != "MANUAL" else "no /l/+vowel resolvable in word window"})
        return {"whisper_text": res["text"].strip(), "all_phones": " ".join(p for p, _ in phones),
                "occurrences": verdicts}


def gate_tensor(res):
    rr = res.retrieval_result
    return None if rr is None else rr.gate_values.detach().cpu().float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", required=True, help="real correction memory (copied, never written)")
    ap.add_argument("--reference", required=True, help="human recording with both pronunciations of 'lead'")
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-corrections", action="store_true", help="reuse <out>/corrections_f5_eval.pt")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "wavs"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(args.out, "run.log")), logging.StreamHandler()])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    disable_spelling_store()
    soundfile_torchaudio_load()

    src_md5 = hashlib.md5(open(args.memory, "rb").read()).hexdigest()
    eval_mem = os.path.join(args.out, "corrections_f5_eval.pt")

    cfg = FlowEditConfig()
    cfg.backbone.backbone_type = "f5tts"
    cfg.memory.retrieval_mode = "baseline"  # correction loop runs exactly as in production
    loop = CorrectionLoop(cfg)
    loop.load_models()
    assert loop.memory.sense_encoder is None and loop.memory.sense_lexicon is None, "semantic models must be off"

    ident = checkpoint_identity(loop.backbone)
    json.dump(ident, open(os.path.join(args.out, "checkpoint_identity.json"), "w"), indent=2)
    logger.info(f"CHECKPOINT IDENTITY: {json.dumps(ident)}")

    import whisper
    asr = whisper.load_model(os.environ.get("CTX_EVAL_WHISPER", "base"))

    # ── Phase 1: real F5 corrections ──
    corr_log = []
    if args.skip_corrections and os.path.exists(eval_mem):
        loop.memory.load(eval_mem)
    else:
        loop.memory.load(args.memory)
        pre = memory_digest(loop.memory)
        clips = crop_reference_clips(asr, args.reference, args.out)
        specs = [
            # carrier sentence, reference clip, user's pronunciation label for the correction
            ("This toy is made up of lead.", clips[0], "led"),
            ("She is the lead of data science.", clips[1], "leed"),
        ]
        for carrier, clip, hint in specs:
            t0 = time.time()
            r = loop.correct(text=carrier, target_word="lead", ref_audio_path=clip["path"],
                             speaker_wav=args.speaker, user_ref_text=clip["transcript"], phonetic_hint=hint)
            opt = r.optimization
            corr_log.append({
                "carrier": carrier, "ref_clip": clip, "phonetic_hint": hint, "success": r.success,
                "error": r.error_message, "memory_index": r.memory_index, "seconds": round(time.time() - t0, 1),
                "token_indices": getattr(r.alignment, "token_indices", None),
                "initial_loss": getattr(opt, "initial_loss", None), "final_loss": getattr(opt, "final_loss", None),
                "word_delta_norm": float(opt.word_delta.norm()) if opt is not None and getattr(opt, "word_delta", None) is not None else None,
            })
            logger.info(f"CORRECTION: {corr_log[-1]}")
        loop.memory.save(eval_mem)
        post = memory_digest(loop.memory)
        corr_log.append({"pre_existing_entries_unchanged":
                         all(a == b for a, b in zip(pre, post[:len(pre)]))})
    json.dump(corr_log, open(os.path.join(args.out, "corrections_log.json"), "w"), indent=2, default=str)
    if corr_log and not all(c.get("success", True) for c in corr_log):
        logger.error("A correction was rejected - see corrections_log.json")

    # ── Phase 2: synthesis, one model instance, all modes ──
    inf = FlowEditInference(cfg)
    inf.load(backbone=loop.backbone, memory=loop.memory)
    mem = inf.memory
    digest_before = memory_digest(mem)
    json.dump(digest_before, open(os.path.join(args.out, "memory_entries.json"), "w"), indent=2)
    cond = inf.backbone.get_speaker_embedding(args.speaker, language="en")
    params = {"seed": SEED, "nfe_step": 32, "cfg_strength": 2.0, "speed": 0.85}

    judge = PhonemeJudge(asr, "cuda" if torch.cuda.is_available() else "cpu")
    rows, diags, inv = [], [], {"baseline_vs_shadow": [], "gates_identical": [], "unrelated_attention": [],
                                "single_sense_baseline_path": []}
    for sid, cat, text, expected in SENTENCES:
        outs = {}
        torch.manual_seed(SEED)
        wav, sr = inf.backbone.synthesize_direct(text=text, speaker_conditioning=cond, language="en", **params)
        p = os.path.join(args.out, "wavs", f"{sid}_rawf5.wav")
        sf.write(p, wav.squeeze().cpu().numpy(), sr)
        outs["rawf5"] = {"wav": p, "audio": wav.squeeze().cpu().numpy()}
        for mode in MODES:
            mem.retrieval_mode = mode
            res = inf.refiner(backbone=inf.backbone, text=text, speaker_conditioning=cond, language="en", **params)
            p = os.path.join(args.out, "wavs", f"{sid}_{mode}.wav")
            sf.write(p, res.waveform.squeeze().cpu().numpy(), res.sample_rate)
            spans = res.diagnostics.get("matched_spans", [])
            outs[mode] = {"wav": p, "audio": res.waveform.squeeze().cpu().numpy(), "gate": gate_tensor(res),
                          "synth_text": res.diagnostics.get("synth_text", text),
                          "delta_norm": res.diagnostics.get("delta_norm", 0.0), "spans": spans,
                          "rerank": list(mem.last_retrieval_diagnostics)}
            diags.append({"id": sid, "mode": mode, "text": text, "expected": expected,
                          "synth_text": outs[mode]["synth_text"], "is_modified": res.is_modified,
                          "gate_max": res.diagnostics.get("gate_max"), "delta_norm": outs[mode]["delta_norm"],
                          "injection_mode": res.diagnostics.get("injection_mode"),
                          "matched_spans": [{**s, "entry_word": mem.entries[mem_i].word if (mem_i := next(
                              (ms.entry_index for ms in res.retrieval_result.matched_spans
                               if ms.char_start == s["char_start"]), -1)) >= 0 else None,
                              "entry_index": mem_i,
                              "entry_carrier": mem.entries[mem_i].carrier_text if mem_i >= 0 else None}
                              for s in spans],
                          "rerank": outs[mode]["rerank"]})

        # invariants
        a, b = outs["baseline"]["audio"], outs["shadow"]["audio"]
        same_len = a.shape == b.shape
        inv["baseline_vs_shadow"].append({"id": sid, "identical": bool(same_len and np.array_equal(a, b)),
                                          "max_abs_diff": float(np.abs(a - b).max()) if same_len else None,
                                          "synth_text_equal": outs["baseline"]["synth_text"] == outs["shadow"]["synth_text"]})
        g = [outs[m]["gate"] for m in MODES]
        inv["gates_identical"].append({"id": sid, "identical": all(x is not None and torch.equal(g[0], x) for x in g)})
        for d in outs["active"]["rerank"]:
            if d.get("applied"):
                inv["unrelated_attention"].append({"id": sid, "char_start": d.get("char_start"),
                                                   "max_other_attn_change": d.get("max_other_attn_change"),
                                                   "group_mass": d.get("group_mass"),
                                                   "group_mass_new": float(sum(d.get("attn_new", [])))})
        if cat.startswith("control"):
            a2 = outs["active"]["audio"]
            inv["single_sense_baseline_path"].append({
                "id": sid, "identical_to_baseline": bool(a.shape == a2.shape and np.array_equal(a, a2)),
                "rerank_applied": any(d.get("applied") for d in outs["active"]["rerank"])})

        # pronunciation scoring
        for mode in ("rawf5",) + MODES:
            j = judge.judge(outs[mode]["wav"], len(expected)) if expected else None
            occ = j["occurrences"] if j else []
            for k, exp in enumerate(expected):
                v = occ[k]
                rows.append({"id": sid, "category": cat, "mode": mode, "text": text, "occurrence": k,
                             "expected": exp, "judged": v["verdict"],
                             "correct": "" if v["verdict"] == "MANUAL" else int(v["verdict"] == exp),
                             "vowel": v.get("vowel"), "window_phones": v.get("window_phones"),
                             "whisper_text": j["whisper_text"], "reason": v.get("reason", ""),
                             "synth_text": outs[mode].get("synth_text", text),
                             "wav": os.path.relpath(outs[mode]["wav"], args.out)})
            if not expected:
                rows.append({"id": sid, "category": cat, "mode": mode, "text": text, "occurrence": "",
                             "expected": "", "judged": "", "correct": "", "vowel": "", "window_phones": "",
                             "whisper_text": "", "reason": "control", "synth_text": outs[mode].get("synth_text", text),
                             "wav": os.path.relpath(outs[mode]["wav"], args.out)})
        logger.info(f"[{sid}] done: " + " | ".join(f"{r['mode']}:{r['judged']}" for r in rows if r["id"] == sid))

    digest_after = memory_digest(mem)
    inv["stored_vectors_unchanged"] = digest_before == digest_after
    inv["source_memory_md5_unchanged"] = hashlib.md5(open(args.memory, "rb").read()).hexdigest() == src_md5
    inv["phoneme_model"] = judge.model_name

    with open(os.path.join(args.out, "pronunciation_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "retrieval_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diags, f, indent=1, default=str, ensure_ascii=False)
    with open(os.path.join(args.out, "retrieval_diagnostics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "mode", "char_start", "entry_index", "entry_word", "phonetic_text", "entry_carrier",
                    "gate", "score", "has_delta", "delta_norm", "synth_text", "rerank_applied", "labels",
                    "ctx_sims", "orig_sims", "attn_base", "attn_new", "baseline_choice", "new_choice"])
        for d in diags:
            rr = {r.get("char_start"): r for r in d["rerank"]}
            for s in d["matched_spans"] or [{}]:
                r = rr.get(s.get("char_start"), {})
                w.writerow([d["id"], d["mode"], s.get("char_start"), s.get("entry_index"), s.get("entry_word"),
                            s.get("phonetic_text"), s.get("entry_carrier"), s.get("gate"), s.get("score"),
                            s.get("has_delta"), d["delta_norm"], d["synth_text"], r.get("applied"),
                            r.get("labels"), r.get("ctx_sims"), r.get("orig_sims"), r.get("attn_base"),
                            r.get("attn_new"), r.get("baseline_choice"), r.get("new_choice")])
    json.dump(inv, open(os.path.join(args.out, "invariants.json"), "w"), indent=2)
    logger.info("DONE")


if __name__ == "__main__":
    main()
