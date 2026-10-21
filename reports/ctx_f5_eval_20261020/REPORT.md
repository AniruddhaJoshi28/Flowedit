# Context-Robust Hopfield Retrieval: real F5-TTS evaluation (2026-10-20)

**Verdict: no measurable improvement.** On 18 automatically scorable target-word occurrences, `active` mode
fixed 1 and broke 2 compared with `baseline`. The pronunciation gains over plain F5-TTS come from the
existing (baseline) retrieval, not from the contextual rerank. Every safety invariant held.

Script: [scripts/ctx_f5_eval.py](../../scripts/ctx_f5_eval.py). Host: iitdserver, RTX A6000 (GPU 0).
Server copy: `~/projects/ctx_eval_20261020` (the production checkout was not touched).

## 1. Setup

| Item | Value |
|---|---|
| Checkpoint | `F5TTS_Base/model_1200000.safetensors`, sha256 `4180310f…7046ae`. This matches the official `SWivid/F5-TTS` file byte-for-byte. |
| Model | `f5_tts.api.F5TTS` → `CFM` → `f5_tts.model.backbones.dit.DiT`, 337.1M params, depth 22, dim 1024 |
| Text embedding | `text_embed` `[2546, 512]`; `encode_text` gives `[1, S, 512]` (character-level), space `F5TTSBackbone:512` |
| Device | `cuda:0`, fp16, torch 2.10.0+cu128 |
| Inference | seed 42, nfe 32, cfg 2.0, speed 0.85, speaker `deploy_voices/blessing.wav`, the same for every mode. Raw F5 and all three modes ran on one model instance. |
| Disabled for the run | MiniLM sense encoder, sense lexicon, and the rule-based spelling store. The first two are external semantic models; the store would respell `lead` by rule. All three were disabled through runtime switches; no code was changed. |
| Not modified | F5 weights, correction optimisation, retrieval code, production `corrections.pt` (MD5 `11958748…46e1` before and after) |

Full identity record: `checkpoint_identity.json`.

## 2. Real correction memory

The existing `corrections.pt` held a single `lead` sense, with 1024-d XTTS-space vectors. A single sense gives
the rerank nothing to choose between. Two corrections were therefore run through the **unmodified
`CorrectionLoop` on F5-TTS**. The reference audio is the human recording `Nice Pronunciation.wav`, cropped by
Whisper word times. The phoneme recogniser confirms the speaker says `l ɛ d` in the first clip and `l iː d` in
the second.

| Carrier | Reference clip | Hint | Loss | Result |
|---|---|---|---|---|
| This toy is made up of lead. | "this toy is made up of lead" | led | 7.82 → 4.53 | Averaged into entry 3 (existing metal sense) |
| She is the lead of data science. | "is the lead of data science" | leed | 10.44 → 7.97 | Inserted as entry 4 |

Each correction took about 372 s. The metal clip aligned with confidence 0.00 (Whisper fallback window), so
its delta may be less precise than the leader clip's (confidence 0.90). Memory used for synthesis:
`corrections_f5_eval.pt`.

## 3. Results

Target words were judged by `facebook/wav2vec2-lv-60-espeak-cv-ft`, a phoneme CTC model with no text prior.
The judge reads the vowel after /l/ inside the Whisper word window. Anything else is marked MANUAL.

| Mode | Correct | Wrong | MANUAL |
|---|---|---|---|
| Raw F5-TTS (no memory) | 8 | 6 | 5 |
| baseline | 17 | 1 (C2) | 1 (B3) |
| shadow | 17 | 1 (C2) | 1 (B3) |
| **active** | **16** | **2 (A2, B4)** | 1 (B3) |

The rerank changed the chosen entry in 4 of 19 occurrences:

| ID | Sentence | Expected | Baseline → active entry | Context similarity [led, leed] | Audio (baseline → active) |
|---|---|---|---|---|---|
| C2 | …Priya was finally promoted to team **lead**. | /liːd/ | led → leed | 0.757, 0.785 | `l ɛ d` → `l iː d` ✅ **fixed** |
| A2 | He was chosen to **lead** the new research project. | /liːd/ | leed → led | 0.882, 0.845 | `l iː d` → `l ɛ d` ❌ **regression** |
| B4 | **Lead** the way, and we will follow you. | /liːd/ | leed → led | 0.919, 0.792 | `l iː d` → `l ɛ d` ❌ **regression** |
| B3 | **Lead** poisoning is a serious health risk. | /lɛd/ | leed → led | 0.921, 0.834 | Unscorable; needs listening |

By category (active vs baseline):

| Category | Effect of active vs baseline |
|---|---|
| Same pronunciation, different neighbours (A1–A6) | 1 regression (A2) |
| Different pronunciation, same spelling (B1–B4) | 1 regression (B4), 1 MANUAL (B3) |
| Long-distance context (C1–C3) | 1 fix (C2) |
| Multiple homographs in one sentence (D1–D3) | No change; all 6 occurrences correct in both modes |

**Failure pattern.** Both regressions are the **verb** "lead" (`to lead the`, `Lead the way`). The memory has
no verb carrier. The context profile is built from F5's character embeddings, so it measures surface
similarity of the neighbouring characters, not meaning. For these short contexts it lands closer to the
metal carrier ("…made of lead") than to the leader carrier ("the lead of data science").

The one fix (C2) won by a margin of 0.028.

## 4. Invariants

All passed. Details are in `invariants.json`.

| Check | Result |
|---|---|
| Stored correction vectors unchanged during synthesis (key, value, word/full delta hashes) | ✅ |
| Production `corrections.pt` unchanged | ✅ |
| Baseline vs shadow waveforms | ✅ 19/19 bit-identical (max abs diff 0.0), identical `synth_text` |
| Gate tensors identical across baseline / shadow / active | ✅ 19/19 |
| Attention outside the `lead` group | ✅ max change 0.0 in all 19 reranks; group mass preserved to ≤1e-7 |
| Single-sense and no-memory sentences (E1–E3) follow the baseline path | ✅ rerank not applied, audio bit-identical |

## 5. Unresolved issues and caveats

- **B3 needs manual listening.** Whisper hears "Bleed" (baseline) and "Blood" (active). The phoneme window
  is empty because the onset is garbled. Raw F5 produces "Lid" with /eɪ/.
- **Raw F5 MANUAL cases.** C1, C3 and both D3 occurrences have unclear vowels (/eɪ/ or a missing /l/). These
  are raw-F5 cases and do not affect the mode comparison.
- **Small sample.** One homograph spelling, one carrier per sense, one speaker, one seed. With 4 decision
  flips, no statistical claim is possible in either direction.
- **Automatic verdicts are not human-verified.** The judge was checked on the human reference recording,
  where it was correct for both senses, but not by listening to the outputs. Listen to the flipped cases
  (A2, B4, C2, B3) before any decision.

## Files

- `wavs/{ID}_{rawf5,baseline,shadow,active}.wav`: 76 files
- `pronunciation_results.csv`: per occurrence and mode; expected vs judged, vowel, phone window, Whisper text
- `retrieval_diagnostics.csv` / `.json`: per span; entry chosen, carrier, phonetic, gate, score, delta norm,
  context similarities, original similarities, attention before and after, baseline and new choice
- `invariants.json`, `checkpoint_identity.json`, `corrections_log.json`, `memory_entries.json`, `run.log`
- `ref_lead_metal.wav`, `ref_lead_leader.wav`: the human reference clips used for the corrections
