# EMA Sense Clusters for Homograph Disambiguation (XTTS + F5-TTS)

## 1. What the XTTS pipeline actually did

The "EMA approach" lives in `HopfieldMemory.write`. A new correction whose context key was
close enough to an existing entry for the same word was merged into it with an exponential moving
average of the key `K`, pooled value `V` and word perturbation `δ*_word`. Otherwise it became a
new entry.

Tracing inference showed that EMA was **not** what disambiguated homographs:

| Step | Mechanism before this change |
|---|---|
| Which entry/sense applies to an occurrence | `HomographContextResolver`: neighbour-word Jaccard, hand-written tense/modal rules |
| Which pronunciation is used | `HOMOGRAPH_SENSE_MAP`: a built-in table of keywords and regexes for 7 words (e.g. `lead`, metal → "led") |
| Whether a word is a homograph | `KNOWN_HOMOGRAPHS`: a fixed list of 20 words, duplicated in memory and refiner |

XTTS sounded right for *"The old paint contains lead, and Priya is the lead researcher"* for two
reasons. The table respelled occurrence 1 as "led". Occurrence 2 was left alone, and the
fine-tuned XTTS GPT, which reads BPE text in context, already says "leed" by default.

### Why F5-TTS failed

1. **Foreign vectors.** F5 loaded XTTS's `corrections.pt` and *linearly interpolated* the 1024-d
   XTTS keys/deltas to 512-d. These vectors mean nothing in F5's character-embedding space.
   Key matching was therefore noise, and the "correction" delta injected into F5 was noise too.
2. **No contextual G2P.** F5 reads characters, so an unedited "lead" gets whatever pronunciation
   the model learned on average. It does not follow the context the way XTTS does.
3. **Shared memory object.** After any correction the API pointed both pipelines at the *same*
   memory object, so XTTS and F5 vectors were cross-interpolated in both directions.
4. **Spelling fallback was per word, not per occurrence.** `apply_corrections_to_text` resolved
   one spelling and substituted it into *every* occurrence. That is exactly the "both
   pronounced identically" symptom.
5. **F5 delta placement** matched only the first 16 characters of the full text. When F5 split
   long text into chunks, the fallback injected the delta at a wrong offset.

## 2. Design

Every memory entry is a **sense cluster**: one pronunciation of one spelling.

```
entry = { word, phonetic_text,                 # pronunciation label (data, never a table)
          sense_prototype P,                   # EMA of contextual occurrence embeddings
          space_data[backbone] = {K, V, δ*},   # per-backbone keys/perturbations
          carrier_texts, access_count }
```

**Contextual sense encoder** (`flowedit/memory/sense_encoder.py`). Each occurrence of a word is
encoded from the middle hidden layers (3–5) of a small pretrained encoder (default
`sentence-transformers/all-MiniLM-L6-v2`). The encoder sees only a ±6-word window that stops at
clause punctuation **and at neighbouring occurrences of the same word**. This is what lets the
two "lead"s of one sentence receive independent context vectors: with whole-sentence encoding,
the second "lead" absorbed "paint contains" and scored 0.84 against the metal sense; with the
bounded window it scores 0.78. The space does not depend on the backbone, so XTTS and F5 share
prototypes.

**Write = EMA clustering.**
- Different known pronunciations are never merged (different sense by definition).
- The same pronunciation is merged.
- When a pronunciation is unknown, merging requires prototype similarity
  ≥ `sense_merge_threshold`. Without an encoder, the backbone-key similarity must reach
  `homograph_sim_threshold`.
- Merge: `P ← norm(ρP + (1-ρ)P_new)`, and the same EMA applies to `K`, `V` and `δ*`, in the local
  backbone space only. ρ = `ema_warmup_decay` while the cluster has ≤ `ema_warmup_count` members,
  then `dedup_ema_decay`.

**Retrieve = per-occurrence assignment.** Each occurrence is assigned to a sense cluster
independently:
- If the word has ≥ 2 senses with different pronunciations, the nearest prototype wins, provided
  it reaches `sense_min_similarity`. This is a relative decision.
- If it has 1 sense, that sense applies only if similarity ≥ `sense_match_threshold`.
- An entry learned without a sentence (carrier == word, e.g. a drug name) applies everywhere.
- Pronunciation comes from the entry, or else from the spelling store resolved for *that*
  occurrence.

**Backbone spaces.** Each backbone has its own `HopfieldMemory` and declares an `embedding_space`.
An entry learned on another backbone is re-keyed with the local `encode_text`, and its foreign δ*
is never injected. On save, every space's tensors are preserved, so F5 can no longer wipe XTTS
deltas and vice versa. The pipelines sync through the shared file (`sync_memories`).

**F5 injection** anchors each corrected span by its own characters plus up to 8 neighbours on
each side. This holds with the reference-prompt prefix, with chunked generation and with
repeated words. If a placement is ambiguous it is skipped rather than guessed.

## 3. Changes

| File | Change |
|---|---|
| `memory/sense_encoder.py` | **new**: windowed contextual occurrence encoder, shared, LRU-cached, thread-safe |
| `memory/hopfield_memory.py` | sense prototypes + EMA; per-occurrence selection; per-backbone `space_data`; re-keying; BPE prefixes decoded once per text (was once per occurrence); `KNOWN_HOMOGRAPHS`, hardcoded EMA warm-up 0.2/0.8 and score scale 10.0 moved to config |
| `alignment/homograph_resolver.py` | `HOMOGRAPH_SENSE_MAP`, tense/modal/pronoun/stop-word tables removed; similarity = contextual embeddings (lexical overlap fallback); `resolve_homograph_phonetic` returns only registered spellings |
| `memory/s3_storage.py` | spellings resolved and substituted **per occurrence** |
| `refiner/hopfield_refiner.py` | duplicated homograph list and resolver re-checks removed; exact offset tracking for respellings; case preserved ("Lead" → "Led"); backbone-declared alignment and safety bounds instead of `dim == 512` and literal clamps |
| `backbone/base.py`, `xtts_wrapper.py`, `f5tts_wrapper.py` | `embedding_space`, `is_character_level`, `delta_bounds` (clamp/cap now in `BackboneConfig`); F5 span-anchored injection; inline preset transcripts replaced by the `voice_finder` registry |
| `pipeline/correction_loop.py`, `pipeline/inference.py` | memory bound to the backbone space/encoder; corrected occurrence's char span stored; encoder warmed at load |
| `api/main.py` | per-backbone memories synced through the file (no object sharing); delete/clear keep F5 in sync |
| `config.py` | all new thresholds and knobs (see §5) |

## 4. Validation

Run anywhere (text level, no TTS weights):

```bash
python scripts/validate_homograph_ema.py --mode text --memory ../corrections.pt \
    --spellings ../spelling_dictionary.json \
    --sense "lead=leed@Priya is the lead researcher on the project"
```

Run on the TTS server (baseline vs corrected WAVs for XTTS and F5 + `manifest.csv`):

```bash
python scripts/validate_homograph_ema.py --mode audio --memory ../corrections.pt \
    --spellings ../spelling_dictionary.json \
    --sense "lead=leed@Priya is the lead researcher on the project" \
    --speaker blessing --out ./homograph_eval
```

Neither mode modifies the production memory file. Tests: `pytest tests` (50 tests, 15 new in
`tests/test_homograph_ema.py`).

### Measured results (text level, real `corrections.pt` + server spelling dictionary)

| Input | XTTS and F5 synthesis text |
|---|---|
| The old paint contains lead, and Priya is the lead researcher studying it. | …contains **led**, and Priya is the **leed** researcher… |
| The lead researcher found lead in the old pipes. | The **leed** researcher found **led** in the old pipes. |
| Lead pipes were banned decades ago. | **Led** pipes were banned… |
| She will lead the team next year. | She will **leed** the team… |
| The weather is pleasant today. | unchanged |

Sense accuracy of the encoder on a held-out probe (5 homographs, 32 occurrences, one example
per sense): 30/32. For the hard `lead` set it is 16/17 with one example per sense and 13/14 with
EMA prototypes built from three corrections per sense.

## 5. Configuration (`MemoryConfig`)

| Key | Default | Meaning |
|---|---|---|
| `sense_encoder_model` | `sentence-transformers/all-MiniLM-L6-v2` (env `FLOWEDIT_SENSE_ENCODER`) | `""` disables; falls back to backbone keys |
| `sense_encoder_layers` | `(3, 4, 5)` | averaged hidden layers |
| `sense_encoder_device` | `cpu` (env `FLOWEDIT_SENSE_DEVICE`) | keeps TTS VRAM untouched |
| `sense_context_words` | 6 | window words on each side |
| `sense_match_threshold` | 0.84 | single learned sense |
| `sense_min_similarity` | 0.60 | floor when several senses compete |
| `sense_merge_threshold` | 0.90 | EMA merge when pronunciation unknown |
| `ema_warmup_count` / `ema_warmup_decay` | 2 / 0.20 | early re-corrections dominate |
| `dedup_ema_decay` | 0.80 | steady-state EMA decay |

## 6. Limitations and trade-offs

- **Pronunciations must exist as data.** With the built-in table gone, a sense is only respelled
  if a correction carries its pronunciation. The current `corrections.pt` `lead` entry stores
  `phonetic_text='lead'` (no information); "led" comes from the spelling dictionary.
- **F5 needs both senses registered.** F5 has no contextual G2P, so an occurrence that matches
  no learned sense is read with F5's default. Register the second sense once (spell mode:
  `POST /api/correct mode=spell text="Priya is the lead researcher" target_word=lead spell_as=leed`).
  XTTS can rely on its own context-aware default for the unregistered sense.
- **A single learned sense is a threshold decision.** Calibrated on 52 held-out occurrences:
  30/31 same-sense applied, 15/21 other-sense rejected at 0.84. Pairs that the encoder barely
  separates (e.g. "bass" fish vs. music, ~0.9) need both senses registered. With two senses the
  decision is relative and much more reliable. More corrections per sense improve the EMA
  prototype.
- **Residual errors** in mixed sentences, e.g. *"Our team lead says the water contains lead"*:
  the noun "team lead" resolves to the metal sense with one leader example. Expected to improve as
  the leader prototype accumulates examples.
- **XTTS-learned δ\* is not used by F5** (and vice versa). Learning a δ\* for F5 requires an
  F5 audio correction. Before, F5 received resampled XTTS vectors, which were noise.
- **Latency.** About 11 ms per *new* sentence that contains a stored word (one CPU encoder pass;
  cached afterwards; zero when no stored word occurs), against 1–3 s of synthesis. One-time
  encoder load is about 2–6 s at pipeline start.
- **Offline servers.** The encoder must be in the HF cache (or `FLOWEDIT_SENSE_ENCODER` must point
  to a local copy). Otherwise the system logs a warning and falls back to backbone context keys,
  which are much weaker for F5's character embeddings.
- **Audio verification** needs a listening pass on the server. ASR transcribes both "led" and
  "leed" as "lead", so it cannot confirm pronunciation automatically.

## 7. Automatic disambiguation: pretrained sense lexicon (no corrections needed)

Learned corrections alone cannot pronounce a homograph the system has never been taught. The
**sense lexicon** (`flowedit/memory/sense_lexicon.py`, data file
`flowedit/resources/sense_lexicon.pt`) adds that knowledge from labelled data. There are no word
lists in code and no LLM.

- **Training data:** Google's Wikipedia Homograph Data (Apache-2.0). It has 16,102 sentences, each
  with one homograph occurrence labelled by sense and IPA pronunciation, covering 162 homographs.
  Build with `python scripts/build_sense_lexicon.py --whd <dir|zip|url>`.
- **Classifier:** one logistic-regression model per homograph. Its input is the occurrence
  embedding (same MiniLM encoder) at two clause-bounded window sizes (3 and 6 words). Each
  occurrence is classified on its own, so one sentence can contain both senses.
  **Held-out accuracy: 96.8% (1,563 / 1,614)**, against 84% for always picking the most frequent
  sense.
- **Pronunciation:** each sense gets a respelling the TTS will read with that pronunciation. Real
  homophones come first ("red", "reed", "led", "tier"). Otherwise the respelling is assembled from
  the most frequent English spellings of the onset and rime ("tare", "wined", "lyve"). Every
  respelling is verified with g2p_en to read back exactly the sense's phones. That gives 41 senses,
  covering the one-syllable homographs. Stress-only pairs (REcord / reCORD, ~266 senses) cannot be
  expressed by spelling and are left as written.
- **Default reading is not touched:** a sense covering ≥ `sense_lexicon_default_share` (0.65) of
  the corpus is the word's normal reading and is not respelled. For example, "lead" = leader
  (79%) is left as "lead", and only the metal sense becomes "led".
- **Learning from corrections:** a spell- or audio-mode correction of a lexicon homograph is
  stored as a labelled context example (`sense_lexicon_user.pt`). A new occurrence whose context is
  close to a corrected example follows the correction. "Close" means word-centred similarity
  ≥ `sense_lexicon_user_radius` (0.75). All other occurrences keep the corpus decision.
  Re-training the classifier on corrections was tried and rejected: it changed 3–6 unrelated
  held-out sentences. Local override changed none.
- **Custom pronunciations:** a correction whose spelling does not denote one of the lexicon's
  pronunciations (e.g. "lehd") becomes that sense's pronunciation for this deployment.

| Check (full pipeline, F5 path) | Result |
|---|---|
| 26 hard sentences (many with two senses in one sentence), no corrections | 20 / 26 sentences, 36 / 42 occurrences |
| same, after 3 one-time corrections | 22 / 26 sentences, 38 / 42 occurrences, no previously-correct output changed |
| "I read books every day, but yesterday I read an entire novel…" | needs one correction ("I read books every day" → reed), then correct |
| "The old paint contains lead, and Priya is the lead researcher…" | correct with no correction |

**Known weak spots:**
- Habitual present tense of *read* in new wordings ("We read the news every morning"): the encoder
  captures meaning better than tense.
- Senses whose cue is distant in a mixed sentence ("caught a bass while … played bass guitar").
- "wound" as a verb.

Part-of-speech tags as extra features were evaluated and rejected: accuracy fell from 96.8% to
95.5% held-out.

**Deployment:**
- `sense_lexicon.pt` ships in git (1.6 MB, un-ignored in `.gitignore`).
- The server needs only the MiniLM encoder, which is already cached there.
- `g2p_en` is optional at runtime. It is only used to map free-form correction spellings to a
  sense.
- The build script additionally needs `nltk` (cmudict) and `g2p_en`.
