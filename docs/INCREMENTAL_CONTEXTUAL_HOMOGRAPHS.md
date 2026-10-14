# Contextual Homograph Handling — Increments over the FlowEdit Paper

**Scope.** This document lists what this repository adds, beyond FlowEdit (arXiv:2606.20518),
to pronounce words whose spelling has several pronunciations. It covers the same word twice in
one sentence with different meanings, tense, noun vs verb, and meaning. For each increment it
gives the problem, the mechanism, the evidence and the location in the code.

**Constraints followed throughout:**
- No hand-written word lists or pronunciation rules in code.
- No large language model.
- No regression for words that need no disambiguation.

Every number below was measured in this repository. Where a number is text-level (the text and
perturbation sent to the TTS) rather than audio-level, it says so.

---

## 0. Baseline: what the paper specifies

| Paper component (Sec. 3, Stage 3) | Specification |
|---|---|
| Memory key / value | `K = pool(c_I)`, `V = pool(δ*_I)` over the corrected token span |
| Retrieval | `Mem(Q) = softmax(βQKᵀ)V`, β = 1/√d, similarity gate σ(max βQKᵀ − τ), τ≈5 |
| Deduplication | cosine > 0.95 → exponential moving average update instead of insertion |
| Capacity | LRU pruning at M_max |
| Homographs | "context-conditioned keys by taking a Gaussian-weighted average of surrounding text embeddings within a window of ±3 tokens" (e.g. *bass* fish vs guitar) |

The paper gives no mechanism for:
- two different meanings of the same spelling in one sentence;
- tense or noun/verb pronunciation changes;
- homographs that were never corrected;
- running one memory across two TTS backbones.

These gaps showed up in practice. *"The old paint contains lead, and Priya is the lead researcher
studying it"* was pronounced with one pronunciation for both occurrences.

---

## 1. Diagnosis of the pre-existing implementation

| # | Defect found | Effect |
|---|---|---|
| D1 | Sense choice came from a hard-coded table (`HOMOGRAPH_SENSE_MAP`, 7 words of keywords/regexes) plus a fixed 20-word `KNOWN_HOMOGRAPHS` list (duplicated in memory and refiner) and tense/modal rules | Not general; the EMA did not actually take part in disambiguation |
| D2 | F5-TTS loaded XTTS memory and **linearly interpolated 1024-d XTTS keys/δ\* to 512-d** | Key matching and injected corrections were noise in F5 space |
| D3 | After any correction both pipelines were pointed at the **same memory object** | XTTS and F5 vectors were cross-interpolated in both directions |
| D4 | Spelling fallback resolved one spelling **per word** and substituted it into every occurrence | Directly caused "both occurrences pronounced the same" |
| D5 | F5 delta placement matched only the first 16 characters of the whole text | Wrong offset when F5 split text into chunks |
| D6 | BPE token prefixes re-decoded **per occurrence** | Avoidable O(n²) cost in XTTS retrieval |
| D7 | Respelling dropped capitalisation ("Lead pipes" → "led pipes") | Prosody/sentence-start errors |

---

## 2. Increments

### 2.1 Per-occurrence contextual sense encoding (replaces the ±3-token Gaussian key)

- **Problem.** The paper's context key averages raw text embeddings of nearby tokens. These are
  character embeddings for F5 and BPE embeddings for XTTS, which carry almost no meaning. When a
  sentence holds both senses, the second occurrence absorbs the first one's context.
- **Mechanism.** Each occurrence is encoded by a pretrained sentence encoder
  (`all-MiniLM-L6-v2`, Apache-2.0), averaging hidden layers 3–5 over the target word's subword
  tokens. The encoder only sees a window of ±N words that **stops at clause punctuation and at the
  neighbouring occurrence of the same word**. The resulting vector is backbone-independent, so XTTS
  and F5 share it.
- **Evidence.**
  - Second "lead" in the target sentence: similarity to the metal sense fell from **0.84
    (whole sentence) to 0.68–0.78 (bounded window)**.
  - Middle layers vs last layer on a 32-occurrence probe across 5 homographs: **30/32 vs 25/32**.
  - Encoder choice: MiniLM layers 3–5 equals DeBERTa-v3-small (29–30/32) at **11 ms vs 310 ms** per
    occurrence on CPU. mpnet-base equals it at 7× the cost.
- **Code.** `flowedit/memory/sense_encoder.py` (`_window`, `encode`, `encode_batch`, shared
  weights, LRU cache, thread-safe).

### 2.2 EMA sense clusters with pronunciation-aware merging (extends the paper's dedup EMA)

- **Problem.** The paper merges on key similarity > 0.95. Two pronunciations of one word in
  similar contexts can merge, which corrupts the stored correction.
- **Mechanism.** Every memory entry is one *pronunciation sense* of a word.
  - Different known pronunciations are never merged. Equal ones are merged.
  - When a pronunciation is unknown, the merge requires prototype similarity
    ≥ `sense_merge_threshold`.
  - The EMA updates the backbone key K, value V, the word perturbation δ\*_word **and a contextual
    sense prototype P**.
  - A configurable warm-up (`ema_warmup_count/decay`) lets early re-corrections dominate.
- **Evidence.**
  - Three corrections per sense, EMA-averaged, raised the hard "lead" set from 10/14 (one example
    per sense) to **13/14**.
  - Unit tests check the EMA arithmetic and that prototypes stay normalised.
- **Code.** `HopfieldMemory.write`, `_find_merge_target`.

### 2.3 Backbone embedding-space isolation (XTTS d=1024 ↔ F5 d=512)

- **Problem.** Defects D2 and D3.
- **Mechanism.**
  - Each backbone declares an `embedding_space` and keeps its own `HopfieldMemory`.
  - Entries keep per-space tensors (`space_data`). An entry learned on another backbone is
    **re-keyed with the local text encoder**, and its foreign δ\* is never injected.
  - Pronunciation and sense prototype transfer unchanged.
  - Pipelines synchronise through the shared file (`sync_memories`). Saving preserves every
    space, so F5 can no longer erase XTTS perturbations or the reverse.
  - Legacy files are detected by key dimension.
- **Evidence.** Round-trip test: XTTS δ\* survives an F5 write. F5 retrieval uses only F5 δ\*. A
  legacy 1024-d file loads into F5 re-keyed with zero foreign δ\*.
- **Code.** `HopfieldMemory._localize`, `_rekey`, `_store_local`, `save/load`;
  `TTSBackbone.embedding_space`; `api/main.py::sync_memories`.

### 2.4 Pronunciation channel: per-occurrence respelling with exact offsets

- **Problem.** The latent δ\* alone cannot carry a pronunciation across backbones (see 2.3), and
  F5 has no contextual grapheme-to-phoneme stage.
- **Mechanism.**
  - Each matched occurrence is respelled independently ("led", "reed").
  - Synthesis-text offsets are computed exactly from the substitutions, with no re-searching, so
    repeated words keep their own positions.
  - Capitalisation follows the original word (fixes D7).
  - Perturbations are aligned per character for F5 and per token for XTTS, using backbone-declared
    bounds instead of `dim == 512` checks and literal clamps.
- **Code.** `HopfieldRefiner._apply_pronunciations`, `forward`.

### 2.5 Span-anchored δ\* injection in F5-TTS

- **Problem.** Defect D5.
- **Mechanism.** Each corrected span is located in the model's input row by its own characters
  plus up to 8 neighbours on each side. Matching tries both F5's +1 vocabulary shift and the raw
  ids. An ambiguous match is skipped rather than guessed. The unconditional CFG row is left
  untouched. Placements are cached per input row across ODE steps.
- **Evidence.** Tests cover the reference-prompt prefix, chunked generation and repeated words.
  The test exposed and fixed a mis-placement where a bare repeated word was matched in the wrong
  chunk.
- **Code.** `F5TTSBackbone._locate_delta_spans`, `_active_delta_spans`, hook in `synthesize_direct`.

### 2.6 Per-occurrence spelling-store resolution

- **Problem.** Defect D4.
- **Mechanism.** Every occurrence is resolved against the original text with its own character
  offset, then substituted right to left. Words decided by the sense lexicon (2.7) are left to it.
- **Code.** `S3SpellingStore.apply_corrections_to_text`.

### 2.7 Pretrained homograph sense lexicon — disambiguation without any correction

- **Problem.** Memory can only pronounce what it was taught. The paper offers nothing for an
  uncorrected homograph.
- **Mechanism.**
  - **Training data:** Wikipedia Homograph Data (Google, Apache-2.0). It has 16,102 sentences
    labelled by sense and IPA pronunciation, covering 162 homographs.
  - **Model:** one logistic-regression classifier per homograph, over occurrence encodings at
    window sizes 3 and 6. Each occurrence is classified on its own.
  - **Respellings, derived from data:**
    - Unambiguous CMU-dictionary homophones are tried first.
    - Otherwise, onset + rime spellings are composed from the most frequent English spellings of
      those sounds (statistics over all CMU monosyllables).
    - Every candidate is **verified with a G2P reader** to read back exactly the sense's phones.
  - **Dominant sense is not respelled:** a sense with ≥ 65% corpus share is the word's default
    reading and is left as written (e.g. "lead" meaning *leader*).
- **Evidence.**
  - **96.8% held-out accuracy (1,563/1,614)**, against 84.0% for always choosing the most frequent
    sense.
  - 41 senses received verified respellings. The other ~266 are stress-only pairs, such as
    REcord / reCORD, which spelling cannot express.
  - 26 hard sentences, most with both senses in one sentence, **no corrections**: 20/26 sentences
    and 36/42 occurrences correct (text level). This includes the reported "lead" sentence.
- **Code.** `flowedit/memory/sense_lexicon.py`, `scripts/build_sense_lexicon.py`, artefact
  `flowedit/resources/sense_lexicon.pt` (1.6 MB, shipped in git).

### 2.8 Learning from corrections by word-centred local override

- **Problem.** Corrections must improve similar sentences without disturbing unrelated ones.
- **Mechanism.**
  - A correction of a lexicon homograph is stored as a labelled context example.
  - Similarity is computed after **subtracting the word's corpus-mean feature**, which removes the
    shared word identity and leaves only the context.
  - An occurrence within `sense_lexicon_user_radius` (0.75) of a corrected example follows it.
    Every other occurrence keeps the corpus decision.
  - A spelling that is none of the lexicon's pronunciations (e.g. "lehd") becomes that sense's
    custom pronunciation.
- **Evidence.**

  | Learning method | Probe occurrences (of 17) | Held-out sentences kept (of 64) |
  |---|---|---|
  | Re-training on corrections (several weightings) | 10–14 | 56–61 |
  | Raw-similarity override | 9–13 | 50–61 |
  | **Word-centred override (shipped)** | **12 → 14** | **64** |

  - End-to-end, 3 corrections raised the hard set from **20/26 to 22/26 sentences with no
    previously-correct output changed**. Both of the user's "read" test sentences became correct
    after one correction.
- **Code.** `SenseLexicon.learn`, `predict`, `_centred`; persisted in `sense_lexicon_user.pt`.

### 2.9 Fine-tuned sense encoder (optional, measured)

- **Mechanism.** The encoder is trained end-to-end on the same corpus, with masked per-word sense
  heads. Its probabilities are averaged with the lexicon's (`sense_model_weight`).
- **Evidence.**
  - Held-out: **98.1% alone, 97.5% averaged**, against 96.8% for the lexicon.
  - Hard probes: 36–39/44, depending on the training run.
  - Individual conversational sentences flipped between training runs. That shows the remaining
    errors come from the data, not the model.
  - Not deployed (`sense_model_weight = 0`): it costs ~3× latency and a 45 MB model for a
    marginal gain, and GlossBERT gives a similar gain more stably.
- **Code.** `flowedit/memory/sense_model.py`, `scripts/train_sense_model.py`.

### 2.10 WordNet word-sense evidence (GlossBERT) — integrated; part-of-speech/tense — rejected by data

- **Motivation.** The corpus is Wikipedia only. It lacks conversational usages ("the actor took a
  bow after the performance", "we read the news every morning") and has no explicit tense signal.
- **Candidates and licences:**

  | Source | Licence | Outcome |
  |---|---|---|
  | ConSeC, ESC, BEM, Babelscape WSL | CC BY-NC / BY-NC-SA | Excluded (non-commercial) |
  | **GlossBERT** (gloss-matching BERT) | MIT | Kept |
  | Self-built definition matching (MiniLM) | Apache-2.0 | Rejected: clearly weaker |
  | spaCy `en_core_web_trf` | — | Not used: its dependency pins older `transformers`, a risk for the F5/XTTS stack |
  | spaCy `en_core_web_md` | MIT | Evaluated |

- **Mechanism.**
  - GlossBERT reads the occurrence in its clause-bounded window and scores every WordNet
    definition of the word.
  - A **WordNet sense → pronunciation table is learned** by running GlossBERT over all 16,102
    labelled corpus sentences and counting, softly, which pronunciation each sense occurs with.
  - The table and the candidate definitions ship inside `sense_lexicon.pt`, so the server needs
    neither WordNet nor NLTK at runtime.
  - Its log-probability is added to the lexicon's with weight 2, and **only when the lexicon is
    less than 95% sure**.
- **Evidence.** Two-fold tuning: weights tuned on one half of the held-out set, accuracy measured
  on the other half.

  | Combination | Held-out accuracy (1,436 multi-sense occurrences) |
  |---|---|
  | Lexicon only | 96.45% |
  | + GlossBERT, always | **97.21%** (folds 97.08% / 97.35%; both chose weight 2) |
  | + GlossBERT, only when lexicon < 95% sure (shipped) | 97.14%, GlossBERT run on 9% of occurrences |
  | + spaCy tag → pronunciation table | no gain: tuning selects weight 0 |

  - **Hard sentences:** unchanged at 20/27 sentences, 37/44 occurrences.
  - GlossBERT now chooses the correct sense for "the actor took a **bow**". With probability 0.52,
    which is below the 0.55 respelling bar, the word is left to the TTS's own reading.
- **Why spaCy was rejected.** The learned tag → pronunciation table is unreliable for exactly the
  hard word. Of corpus occurrences of *read* that spaCy tagged present tense (VBP), 4 of 5 were
  labelled past ("my parents never read a book"). For an unchanged spelling the tagger infers tense
  from the same context the classifier already uses. Forcing "present tag → reed" would be a
  hand-written rule, so it was not done.
- **Why the bow gesture sense stays hard.** WordNet has the right definition (*bow.n.07*,
  acknowledging applause), but it occurs only twice in the labelled corpus. The learned table
  therefore carries little evidence for it.
- **Cost.** GlossBERT is BERT-base, 406 MB on disk. It costs about 300 ms per CPU sentence that
  contains an uncertain homograph, and a few milliseconds on GPU.
- **Code.** `flowedit/memory/gloss_wsd.py`, `SenseLexicon._wsd_probs`,
  `scripts/precompute_wsd.py`, `scripts/build_wsd_tables.py`. Weights go in `models/glossbert/`
  (git-ignored).

---

## 3. Hard-coding removed

| Removed | Replaced by |
|---|---|
| `HOMOGRAPH_SENSE_MAP` (keywords + regex for 7 words), `KNOWN_HOMOGRAPHS` (×2), tense/modal/pronoun/stop-word tables in the resolver | Learned context encoders, corpus-trained classifiers, user data |
| Inline EMA warm-up 0.2/0.8, score scale 10.0, gate floor 0.5 | `MemoryConfig` fields |
| Literal δ clamps 0.30/0.35 and norm caps 4.0/3.5 in wrappers and refiner | `BackboneConfig` + `TTSBackbone.delta_bounds` |
| `backbone_dim == 512` checks to detect F5 | `TTSBackbone.is_character_level` |
| Inline Blessing/Michael transcripts in the F5 wrapper | `voice_finder` registry (behaviour-preserving) |

Not changed, and out of homograph scope, because they could not be validated without audio:
- preset-voice trim timings;
- absolute deployment paths;
- the API port list;
- the stop-word list used by ASR transcript clean-up.

---

## 4. Validation summary

| Check | Result |
|---|---|
| Unit/integration tests | 55 pass (35 pre-existing + 20 new) |
| Sense lexicon, held-out | 96.8% (fine-tuned 98.1%, average 97.5%); lexicon + GlossBERT 97.2% (fair 2-fold) |
| Hard mixed-sense sentences, no corrections | 20/26 sentences, 36/42 occurrences (27-sentence set incl. the reported bow sentence: 20/27, 37/44) |
| Same, after 3 corrections | 22/26, 38/42, zero collateral changes |
| Server log, reported sentence | "contains **led** … the **lead** researcher" decided per occurrence |
| Latency | ~10–30 ms per new sentence containing a homograph (CPU encoder); 0 ms otherwise |

All accuracy figures are text-level, i.e. the decision and respelling sent to the TTS. Audio was
checked by listening on the GPU server. ASR cannot verify it automatically, because "led" and
"leed" are both transcribed as "lead".

---

## 5. Known limitations

- **Conversational tense** for *read* in phrasings absent from the corpus. Mitigated by one
  correction for that phrasing, and by the tense signal in 2.10.
- **Two cues in one clause-free sentence** ("caught a bass while … played bass guitar").
- **Stress-only homographs** (record, present, object…) are left to the TTS, because a respelling
  cannot carry stress.
- A word with a single learned sense and no lexicon coverage relies on a similarity threshold
  (0.84). On held-out contexts: 30/31 same-sense applied, 15/21 other-sense rejected.

---

## 6. Negative results (tried and rejected)

| Attempt | Outcome |
|---|---|
| Whole-sentence occurrence encoding | Second occurrence contaminated (0.84 vs 0.68–0.78 bounded) |
| Rule-free POS one-hot features from NLTK | Held-out fell to 95.5% |
| spaCy tag → pronunciation table (learned) | Tuned weight 0; *read* tagged VBP was labelled past in 4 of 5 cases |
| Fine-tuned encoder as the main classifier | +1.3 held-out points, but individual conversational sentences flipped between training runs |
| g2p_en homograph table as primary mechanism | 13/23; lacks lead/bow/tear/wind/live/bass; *read* entry inverted |
| Re-training the classifier on corrections | Broke 3–6 unrelated held-out sentences |
| Raw-similarity correction override | Up to 16 held-out sentences changed |
| Larger encoders (mpnet, DeBERTa) | No accuracy gain, 7–30× slower |
| Bare-word baseline as single-sense threshold | Uninformative (bare "lead" ≈ metal context) |

---

## 7. Deployment

| Component | Where | In git |
|---|---|---|
| Sense lexicon (classifiers, respellings, WordNet tables) | `flowedit/resources/sense_lexicon.pt` (1.8 MB) | yes |
| Contextual encoder | `sentence-transformers/all-MiniLM-L6-v2` (HF cache) | no (cached on server) |
| GlossBERT weights | `models/glossbert/` (`config.json`, `pytorch_model.bin`, `vocab.txt`) | no (copy once) |
| Learned corrections | `./sense_lexicon_user.pt`, `./corrections.pt` | runtime data |

- If GlossBERT is missing, the system logs a warning and runs on the lexicon alone.
- Rebuilding from scratch, in order:
  1. `scripts/build_sense_lexicon.py --whd …`
  2. `scripts/precompute_wsd.py --whd … --glossbert models/glossbert` (GPU)
  3. `scripts/build_wsd_tables.py --scores glossbert_whd.pt`
