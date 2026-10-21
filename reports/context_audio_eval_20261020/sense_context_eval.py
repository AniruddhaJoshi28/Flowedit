#!/usr/bin/env python
"""
Real F5-TTS audio evaluation of the context sense model (flowedit/memory/sense_context.py)
against the original pooled-key sense choice (FLOWEDIT_SENSE_CONTEXT=0).

  0. Checkpoint identity (genuine F5-TTS, sha256, shapes, device).
  1. References: each sense of each homograph gets 3 correction sentences. Human clips are used where
     available (Nice Pronunciation.wav: "made up of lead" / "the lead of data science"); the rest are
     F5-TTS readings of the respelled carrier in a different voice. Every reference must show the
     intended vowel in a phoneme recogniser before it is used.
  2. Corrections: the unmodified CorrectionLoop on F5-TTS writes every correction (real δ*).
  3. Text-level retrieval: the real retrieval path on held-out sentences, pooled vs context.
  4. Audio: raw F5, pooled, context on every held-out sentence (same model, seed, speaker), judged by
     a phoneme CTC model inside Whisper word windows; unclear cases are marked MANUAL.
  Invariants: stored vectors unchanged, gate tensors identical across modes, words with a single
  learned pronunciation unaffected.

The rule-based spelling store is disabled so only the memory decides. With --skip-corrections an
existing evaluation memory is reused and only corrections whose sentence it lacks are run.
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

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flowedit.config import FlowEditConfig  # noqa: E402
from flowedit.memory.s3_storage import s3_spelling_store  # noqa: E402
from flowedit.pipeline.correction_loop import CorrectionLoop  # noqa: E402
from flowedit.pipeline.inference import FlowEditInference  # noqa: E402

logger = logging.getLogger("sense_context_eval")
SEED = 42
PARAMS = {"seed": SEED, "nfe_step": 32, "cfg_strength": 2.0, "speed": 0.85}

IY, EH = "IY", "EH"   # /iː/ (leed, reed) vs /ɛ/ (led, red)
UW, AW = "UW", "AW"   # /uː/ (woond, injury) vs /aʊ/ (wownd, past of wind)
SPELL = {("lead", EH): "led", ("lead", IY): "leed", ("read", EH): "red", ("read", IY): "reed",
         ("wound", UW): "woond", ("wound", AW): "wownd"}

# (word, vowel, carrier, human clip name or None)
CORRECTIONS = [
    ("lead", EH, "This toy is made up of lead.", "metal"),
    ("lead", EH, "The old pipes were full of lead.", None),
    ("lead", EH, "Lead paint is dangerous for children.", None),
    ("lead", IY, "She is the lead of data science.", "leader"),
    ("lead", IY, "He will lead the team next year.", None),
    ("lead", IY, "Our team lead approved the plan.", None),
    ("read", EH, "Yesterday she read the whole book.", None),
    ("read", EH, "I have already read that report.", None),
    ("read", EH, "He read the letter twice last night.", None),
    ("read", IY, "Please read the instructions carefully.", None),
    ("read", IY, "I like to read before bed.", None),
    ("read", IY, "Can you read this sentence aloud?", None),
    ("wound", UW, "The nurse cleaned the wound carefully.", None),
    ("wound", UW, "The soldier had a deep wound in his leg.", None),
    ("wound", UW, "Words can wound people deeply.", None),
    ("wound", AW, "She wound the clock before bed.", None),
    ("wound", AW, "He wound the rope around the post.", None),
    ("wound", AW, "The path wound through the hills.", None),
]

# Held-out sentences: (id, category, text, [(word, expected vowel) in order of appearance])
TESTS = [
    ("L01", "lead_metal", "The pipes in the basement are made of lead.", [("lead", EH)]),
    ("L02", "lead_metal", "Lead is a soft and heavy metal.", [("lead", EH)]),
    ("L03", "lead_metal", "They found lead in the drinking water.", [("lead", EH)]),
    ("L04", "lead_metal", "The bullet was made of lead and copper.", [("lead", EH)]),
    ("L05", "lead_metal", "Old paint often contains lead.", [("lead", EH)]),
    ("L06", "lead_metal", "Exposure to lead harms the nervous system.", [("lead", EH)]),
    ("L07", "lead_metal", "The fishing weights were cast from solid lead.", [("lead", EH)]),
    ("L08", "lead_metal", "A thin sheet of lead blocks the radiation.", [("lead", EH)]),
    ("L09", "lead_metal", "The miners extracted lead and zinc from the ore.", [("lead", EH)]),
    ("L10", "lead_metal", "Children should never play with lead toys.", [("lead", EH)]),
    ("L11", "lead_guide", "She will lead the meeting tomorrow.", [("lead", IY)]),
    ("L12", "lead_guide", "He was asked to lead the project.", [("lead", IY)]),
    ("L13", "lead_guide", "Who will lead the team after she retires?", [("lead", IY)]),
    ("L14", "lead_guide", "The lead singer walked onto the stage.", [("lead", IY)]),
    ("L15", "lead_guide", "Our team lead scheduled a review.", [("lead", IY)]),
    ("L16", "lead_guide", "The guide will lead us through the forest.", [("lead", IY)]),
    ("L17", "lead_guide", "She took the lead in the final lap.", [("lead", IY)]),
    ("L18", "lead_guide", "Priya is the lead engineer on this product.", [("lead", IY)]),
    ("L19", "lead_guide", "These roads lead to the old city.", [("lead", IY)]),
    ("L20", "lead_guide", "Let me lead the discussion today.", [("lead", IY)]),
    ("R01", "read_past", "She read the report last night.", [("read", EH)]),
    ("R02", "read_past", "I read that book when I was a child.", [("read", EH)]),
    ("R03", "read_past", "He had already read the email before the meeting.", [("read", EH)]),
    ("R04", "read_past", "Yesterday we read three chapters together.", [("read", EH)]),
    ("R05", "read_past", "Have you read the new policy?", [("read", EH)]),
    ("R06", "read_past", "They read the results and smiled.", [("read", EH)]),
    ("R07", "read_past", "The letter was read aloud at the ceremony.", [("read", EH)]),
    ("R08", "read_past", "I read about the accident in the newspaper last week.", [("read", EH)]),
    ("R09", "read_past", "She read every page twice before signing.", [("read", EH)]),
    ("R10", "read_past", "Last year he read over fifty novels.", [("read", EH)]),
    ("R11", "read_present", "I read the news every morning.", [("read", IY)]),
    ("R12", "read_present", "Please read the label before use.", [("read", IY)]),
    ("R13", "read_present", "Children love to read comic books.", [("read", IY)]),
    ("R14", "read_present", "Can you read the second paragraph?", [("read", IY)]),
    ("R15", "read_present", "I will read your proposal tomorrow.", [("read", IY)]),
    ("R16", "read_present", "She likes to read in the garden.", [("read", IY)]),
    ("R17", "read_present", "Read the question carefully before answering.", [("read", IY)]),
    ("R18", "read_present", "We read together every evening.", [("read", IY)]),
    ("R19", "read_present", "It is hard to read without my glasses.", [("read", IY)]),
    ("R20", "read_present", "They want to read the contract first.", [("read", IY)]),
    ("X01", "long_distance", "After hours of digging through the ruins of the old factory, the archaeologists "
                             "finally identified the grey material as lead.", [("lead", EH)]),
    ("X02", "long_distance", "After many years of loyal service and countless successful projects, Maria was "
                             "chosen to lead the company.", [("lead", IY)]),
    ("X03", "long_distance", "When I was a student living in a tiny apartment far from home, I read every "
                             "book in the library.", [("read", EH)]),
    ("X04", "long_distance", "Every night before going to sleep, no matter how tired I am, I like to read a "
                             "few pages of a novel.", [("read", IY)]),
    ("M01", "multi_homograph", "Yesterday I read the report about lead in the water.", [("read", EH), ("lead", EH)]),
    ("M02", "multi_homograph", "She will lead the team and read the results to everyone.", [("lead", IY), ("read", IY)]),
    ("M03", "multi_homograph", "The lead researcher found lead in the old pipes.", [("lead", IY), ("lead", EH)]),
    ("M04", "multi_homograph", "Please read the warning: this paint contains lead.", [("read", IY), ("lead", EH)]),
    ("W01", "wound_injury", "The wound healed after two weeks.", [("wound", UW)]),
    ("W02", "wound_injury", "She pressed a bandage on the wound.", [("wound", UW)]),
    ("W03", "wound_injury", "A small wound can still get infected.", [("wound", UW)]),
    ("W04", "wound_injury", "The doctor stitched the open wound.", [("wound", UW)]),
    ("W05", "wound_injury", "Keep the wound clean and dry.", [("wound", UW)]),
    ("W06", "wound_wind", "They wound the cable onto the spool.", [("wound", AW)]),
    ("W07", "wound_wind", "I wound the string around my finger.", [("wound", AW)]),
    ("W08", "wound_wind", "The river wound slowly to the sea.", [("wound", AW)]),
    ("W09", "wound_wind", "We wound the yarn into a ball.", [("wound", AW)]),
    ("W10", "wound_wind", "Grandpa wound his watch every morning.", [("wound", AW)]),
    ("C01", "control_single", "Bispecific antibodies are becoming an important approach in cancer treatment.", []),
    ("C02", "control_none", "The weather is pleasant today.", []),
]

MODES = {"pooled": False, "context": True}   # name -> MemoryConfig.sense_context

# Phoneme-recogniser vowel symbols for each pronunciation class (evaluation only)
VOWELS = {IY: {"iː", "i", "ɪː", "iə"}, EH: {"ɛ", "e", "ɛː", "æ", "ɛə"},
          UW: {"uː", "u", "ʊ", "ʉ", "ʉː"}, AW: {"aʊ", "aː", "a", "ɑ", "æʊ"}}
CLASSES = {"lead": (IY, EH), "read": (IY, EH), "wound": (UW, AW)}
WORD_RE = {"lead": re.compile(r"^(lead|led|leed|lid|lied|lede|leads)$"),
           "read": re.compile(r"^(read|red|reed|reid|rid|reads)$"),
           "wound": re.compile(r"^(wound|woond|wownd|wand|wind|wounded|round|womb)$")}
ONSET = {"lead": {"l"}, "read": {"ɹ", "r", "ɾ"}, "wound": {"w"}}


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def tensor_digest(t):
    if t is None:
        return None
    return hashlib.sha256(t.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()[:16]


def memory_digest(mem):
    return [{"idx": i, "word": e.word, "phonetic": e.phonetic_text, "carriers": list(e.carrier_texts or []),
             "key": tensor_digest(e.key), "value": tensor_digest(e.value),
             "word_delta": tensor_digest(e.word_delta), "full_delta": tensor_digest(e.full_delta)}
            for i, e in enumerate(mem.entries)]


def isolate_runtime():
    """No rule-based respelling; decode WAV without torchcodec (broken FFmpeg linkage on this host)."""
    import torchaudio
    s3_spelling_store._dictionary = {}
    s3_spelling_store._ensure_loaded = lambda: None
    s3_spelling_store._save_local_cache = lambda: False

    def _load(path, *args, **kwargs):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sr

    torchaudio.load = _load


def set_mode(mem, name):
    mem.config.sense_context = MODES[name]


def vowel_of(phonetic, word):
    p = (phonetic or "").strip().lower()
    for (w, v), s in SPELL.items():
        if w == word and p == s:
            return v
    return None


def checkpoint_identity(backbone):
    api = backbone.tts_api
    if api is None:
        raise RuntimeError("F5-TTS API not loaded (stub fallback)")
    ema = api.ema_model
    tr = ema.transformer
    ckpt = getattr(api, "ckpt_file", None) or backbone._resolve_f5_paths()[0]
    info = {
        "backbone_class": type(backbone).__name__,
        "transformer_class": f"{type(tr).__module__}.{type(tr).__name__}",
        "checkpoint": os.path.abspath(ckpt), "checkpoint_sha256": sha256(ckpt),
        "params_total": sum(p.numel() for p in ema.parameters()),
        "text_embed_weight": list(tr.text_embed.text_embed.weight.shape),
        "encode_text_shape": list(backbone.encode_text("She will read it.", "en").shape),
        "embedding_space": backbone.embedding_space,
        "device": str(next(ema.parameters()).device), "dtype": str(next(ema.parameters()).dtype),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if "xtts" in info["backbone_class"].lower() or info["params_total"] < 100e6:
        raise RuntimeError(f"Not genuine F5-TTS: {info}")
    return info


class PhonemeJudge:
    """Phoneme CTC (wav2vec2 espeak, no text prior) with frame timing + Whisper word windows."""

    def __init__(self, whisper_model, device):
        from transformers import AutoModelForCTC, AutoProcessor
        self.name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
        self.proc = AutoProcessor.from_pretrained(self.name)
        self.model = AutoModelForCTC.from_pretrained(self.name).to(device).eval()
        self.device = device
        self.whisper = whisper_model

    def phones(self, y16):
        with torch.no_grad():
            inp = self.proc(y16, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
            ids = self.model(inp).logits[0].argmax(-1).cpu().tolist()
        step = len(y16) / 16000 / max(1, len(ids))
        pad, out, prev = self.proc.tokenizer.pad_token_id, [], None
        for k, i in enumerate(ids):
            if i != prev and i != pad:
                out.append((self.proc.tokenizer.convert_ids_to_tokens(i), k * step))
            prev = i
        return out

    def judge(self, wav_path, expected):
        """expected: [(word, vowel)] in order. Returns per-occurrence verdict IY / EH / MANUAL."""
        import librosa
        y16, _ = librosa.load(wav_path, sr=16000)
        ph = self.phones(y16)
        res = self.whisper.transcribe(wav_path, word_timestamps=True, language="en")
        words = [w for seg in res["segments"] for w in seg["words"]]
        out = []
        seen = {}
        for word, _ in expected:
            k = seen.get(word, 0)
            seen[word] = k + 1
            cands = [w for w in words if WORD_RE[word].match(re.sub(r"[^a-z']", "", w["word"].lower()))]
            n_exp = sum(1 for wd, _ in expected if wd == word)
            if len(cands) != n_exp:
                out.append({"verdict": "MANUAL", "reason": f"whisper found {len(cands)} {word}-like words",
                            "window": ""})
                continue
            w = cands[k]
            win = [p for p, t in ph if w["start"] - 0.10 <= t <= w["end"] + 0.10]
            vowel = next((win[j + 1] for j in range(len(win) - 1) if win[j] in ONSET[word]), None)
            v = next((c for c in CLASSES[word] if vowel in VOWELS[c]), "MANUAL")
            out.append({"verdict": v, "vowel": vowel, "window": " ".join(win),
                        "reason": "" if v != "MANUAL" else "no onset+vowel in word window"})
        return {"whisper": res["text"].strip(), "occ": out}


def text_level(inf, mem):
    """Real retrieval path (F5 text encoder) per held-out occurrence, without synthesis."""
    rows = []
    tok = getattr(inf.backbone, "tokenizer_instance", None)
    for sid, cat, text, expected in TESTS:
        if not expected:
            continue
        q = inf.backbone.encode_text(text, "en")
        for name in MODES:
            set_mode(mem, name)
            r = mem.retrieve(q, text=text, tokenizer=tok, language="en")
            spans = sorted(r.matched_spans, key=lambda s: s.char_start)
            seen = {}
            for word, exp in expected:
                k = seen.get(word, 0)
                seen[word] = k + 1
                occ = [s for s in spans if s.word.lower() == word]
                got = vowel_of(occ[k].phonetic_text, word) if k < len(occ) else None
                rows.append({"id": sid, "category": cat, "config": name, "word": word, "occurrence": k,
                             "expected": exp, "chosen": got or "ABSTAIN", "correct": int(got == exp) if got else ""})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", required=True, help="real correction memory (copied, never written)")
    ap.add_argument("--reference", required=True, help="human recording with both pronunciations of 'lead'")
    ap.add_argument("--speaker", required=True, help="voice used for every evaluated synthesis")
    ap.add_argument("--ref-speaker", required=True, help="different voice for F5-made correction references")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-corrections", action="store_true", help="reuse <out>/corrections_eval.pt")
    args = ap.parse_args()

    out = args.out
    for d in ("wavs", "refs"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(out, "run.log")), logging.StreamHandler()])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    isolate_runtime()
    src_md5 = hashlib.md5(open(args.memory, "rb").read()).hexdigest()
    eval_mem = os.path.join(out, "corrections_eval.pt")

    cfg = FlowEditConfig()
    cfg.backbone.backbone_type = "f5tts"
    loop = CorrectionLoop(cfg)
    loop.load_models()
    ident = checkpoint_identity(loop.backbone)
    json.dump(ident, open(os.path.join(out, "checkpoint_identity.json"), "w"), indent=2)
    logger.info(f"CHECKPOINT IDENTITY: {json.dumps(ident)}")

    import whisper
    asr = whisper.load_model(os.environ.get("EVAL_WHISPER", "medium"))
    judge = PhonemeJudge(asr, "cuda" if torch.cuda.is_available() else "cpu")

    # ── Phase 1+2: references and real corrections ──
    corr_log = []
    import librosa
    reuse = args.skip_corrections and os.path.exists(eval_mem)
    loop.memory.load(eval_mem if reuse else args.memory)
    done = {c for e in loop.memory.entries for c in (e.carrier_texts or [])} if reuse else set()
    if True:
        y, sr = librosa.load(args.reference, sr=24000)
        words = [w for s in asr.transcribe(args.reference, word_timestamps=True)["segments"] for w in s["words"]]
        toks = [re.sub(r"[^a-z']", "", w["word"].lower()) for w in words]
        leads = [i for i, t in enumerate(toks) if t == "lead"]
        first, second = leads[0], leads[1]
        a1 = max(i for i in range(first) if toks[i] == "this")
        a2 = second - 2 if toks[second - 2] == "is" else second - 1
        b2 = next((i for i in range(second, len(toks)) if toks[i] == "science"), min(second + 4, len(toks) - 1))
        human = {}
        for name, a, b in (("metal", a1, first), ("leader", a2, b2)):
            t0, t1 = max(0.0, words[a]["start"] - 0.10), words[b]["end"] + 0.15
            p = os.path.join(out, "refs", f"human_{name}.wav")
            sf.write(p, y[int(t0 * sr):int(t1 * sr)], sr)
            human[name] = (p, " ".join(toks[a:b + 1]))

        ref_cond = loop.backbone.get_speaker_embedding(args.ref_speaker, language="en")
        for n, (word, vowel, carrier, clip) in enumerate(CORRECTIONS):
            if carrier in done:
                corr_log.append({"word": word, "carrier": carrier, "reused": True})
                continue
            spelling = SPELL[(word, vowel)]
            if clip:
                ref_path, ref_text = human[clip]
                source = "human"
            else:
                respelled = re.sub(rf"\b{word}\b", lambda m: spelling.capitalize() if m.group()[0].isupper() else spelling,
                                   carrier, count=1, flags=re.IGNORECASE)
                wav, sr_ = loop.backbone.synthesize_direct(text=respelled, speaker_conditioning=ref_cond,
                                                           language="en", **{**PARAMS, "seed": 7})
                ref_path = os.path.join(out, "refs", f"f5_{n:02d}_{spelling}.wav")
                sf.write(ref_path, wav.squeeze().cpu().numpy(), sr_)
                ref_text, source = carrier.rstrip(".?!").lower(), f"f5:{respelled}"
            check = judge.judge(ref_path, [(word, vowel)])["occ"][0]
            entry = {"word": word, "expected_vowel": vowel, "carrier": carrier, "hint": spelling,
                     "reference": os.path.relpath(ref_path, out), "source": source, "reference_check": check}
            if check["verdict"] != vowel:
                entry.update({"success": False, "error": "reference does not carry the intended vowel; skipped"})
                corr_log.append(entry)
                logger.warning(f"REFERENCE REJECTED: {entry}")
                continue
            t0 = time.time()
            r = loop.correct(text=carrier, target_word=word, ref_audio_path=ref_path, speaker_wav=args.speaker,
                             user_ref_text=ref_text, phonetic_hint=spelling)
            opt = r.optimization
            entry.update({"success": r.success, "error": r.error_message, "memory_index": r.memory_index,
                          "seconds": round(time.time() - t0, 1),
                          "align_conf": getattr(r.alignment, "confidence", None),
                          "initial_loss": getattr(opt, "initial_loss", None),
                          "final_loss": getattr(opt, "final_loss", None)})
            corr_log.append(entry)
            logger.info(f"CORRECTION: {entry}")
        loop.memory.save(eval_mem)
    json.dump(corr_log, open(os.path.join(out, "corrections_log.json"), "w"), indent=2, default=str)

    inf = FlowEditInference(cfg)
    inf.load(backbone=loop.backbone, memory=loop.memory)
    mem = inf.memory
    before = memory_digest(mem)
    json.dump(before, open(os.path.join(out, "memory_entries.json"), "w"), indent=2)

    # ── Phase 3: text-level retrieval ──
    trows = text_level(inf, mem)
    with open(os.path.join(out, "text_level_selection.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trows[0].keys()))
        w.writeheader()
        w.writerows(trows)
    summary = {}
    for r in trows:
        s = summary.setdefault(r["config"], {"correct": 0, "wrong": 0, "abstain": 0})
        s["abstain" if r["chosen"] == "ABSTAIN" else ("correct" if r["correct"] == 1 else "wrong")] += 1
    json.dump(summary, open(os.path.join(out, "text_level_summary.json"), "w"), indent=2)
    logger.info(f"TEXT-LEVEL: {summary}")

    # ── Phase 4: audio ──
    cond = inf.backbone.get_speaker_embedding(args.speaker, language="en")
    rows, diags = [], []
    inv = {"gates_identical": [], "single_sense_identical": []}
    for sid, cat, text, expected in TESTS:
        audio, gates = {}, {}
        torch.manual_seed(SEED)
        wav, sr = inf.backbone.synthesize_direct(text=text, speaker_conditioning=cond, language="en", **PARAMS)
        p = os.path.join(out, "wavs", f"{sid}_rawf5.wav")
        sf.write(p, wav.squeeze().cpu().numpy(), sr)
        audio["rawf5"] = (p, wav.squeeze().cpu().numpy(), text)
        for mode in MODES:
            set_mode(mem, mode)
            res = inf.refiner(backbone=inf.backbone, text=text, speaker_conditioning=cond, language="en", **PARAMS)
            p = os.path.join(out, "wavs", f"{sid}_{mode}.wav")
            a = res.waveform.squeeze().cpu().numpy()
            sf.write(p, a, res.sample_rate)
            synth = res.diagnostics.get("synth_text", text)
            audio[mode] = (p, a, synth)
            gates[mode] = res.retrieval_result.gate_values.detach().cpu().float()
            diags.append({"id": sid, "mode": mode, "text": text, "synth_text": synth,
                          "gate_max": res.diagnostics.get("gate_max"), "delta_norm": res.diagnostics.get("delta_norm"),
                          "matched_spans": [{**s, "entry_carriers": mem.entries[ms.entry_index].carrier_texts}
                                            for s, ms in zip(res.diagnostics.get("matched_spans", []),
                                                             res.retrieval_result.matched_spans)],
                          "sense_decisions": list(mem.last_sense_decisions)})
        inv["gates_identical"].append({"id": sid, "identical": torch.equal(gates["pooled"], gates["context"])})
        if cat.startswith("control"):
            a, b = audio["pooled"][1], audio["context"][1]
            inv["single_sense_identical"].append({"id": sid, "identical": bool(a.shape == b.shape and np.array_equal(a, b))})
        for mode, (p, _, synth) in audio.items():
            if not expected:
                rows.append({"id": sid, "category": cat, "mode": mode, "text": text, "word": "", "occurrence": "",
                             "expected": "", "judged": "", "correct": "", "vowel": "", "window": "",
                             "whisper": "", "synth_text": synth, "wav": os.path.relpath(p, out)})
                continue
            j = judge.judge(p, expected)
            for k, ((word, exp), v) in enumerate(zip(expected, j["occ"])):
                rows.append({"id": sid, "category": cat, "mode": mode, "text": text, "word": word, "occurrence": k,
                             "expected": exp, "judged": v["verdict"],
                             "correct": "" if v["verdict"] == "MANUAL" else int(v["verdict"] == exp),
                             "vowel": v.get("vowel"), "window": v.get("window"), "whisper": j["whisper"],
                             "synth_text": synth, "wav": os.path.relpath(p, out)})
        logger.info(f"[{sid}] " + " | ".join(f"{r['mode']}:{r['judged']}" for r in rows if r["id"] == sid))

    inv["stored_vectors_unchanged"] = memory_digest(mem) == before
    inv["source_memory_md5_unchanged"] = hashlib.md5(open(args.memory, "rb").read()).hexdigest() == src_md5
    with open(os.path.join(out, "pronunciation_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "retrieval_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diags, f, indent=1, default=str, ensure_ascii=False)
    json.dump(inv, open(os.path.join(out, "invariants.json"), "w"), indent=2)
    logger.info("DONE")


if __name__ == "__main__":
    main()
