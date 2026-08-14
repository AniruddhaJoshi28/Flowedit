# FlowEdit Core Speech Generation Audit Report

**Date:** 2026-08-03  
**Branch:** `feature/new_models`  
**Repository:** FlowEdit Speech Generation Core  

## Executive Summary

A comprehensive research-grade audit was performed on the FlowEdit codebase across all supported model backbones (F5-TTS, CosyVoice 2, and XTTS-v2). The audit confirmed several critical flaws in inference routing, reference transcript handling, latent optimization, audio preprocessing, and memory isolation.

---

## Confirmed Issues & Audit Details

### 1. Silent Audio Fallback and Unvalidated Reference Audio
- **File Path:** `flowedit/backbone/cosyvoice_wrapper.py`, `flowedit/backbone/f5tts_wrapper.py`, `flowedit/utils/audio.py`
- **Functions:** `get_speaker_embedding()`, `synthesize_from_embeddings()`, `load_audio()`
- **Problem:** When reference audio is missing, invalid, or corrupted, the code silently loads default demo WAVs (including hardcoded paths like `/home/rsurya/...` or bundled Chinese prompt audio).
- **Consequence:** Users receive audio synthesized with arbitrary voices without warning; invalid inputs pass silently.
- **Fix Applied:** Implement `flowedit/audio/audio_preprocessor.py` and `flowedit/audio/prompt_validator.py`. Raise `ReferenceAudioError` on invalid reference audio and completely remove silent demo/default voice loading when speaker audio is supplied.

### 2. Fake / Dummy Optimization Loss & Sine Wave Fallback
- **File Path:** `flowedit/backbone/cosyvoice_wrapper.py`
- **Functions:** `compute_optimization_loss()`, `synthesize_from_embeddings()`
- **Problem:** `compute_optimization_loss()` calculates `F.mse_loss(perturbed_embeddings, base_embeddings)`, which collapses the learned perturbation $\delta$ toward zero instead of evaluating speech output. `synthesize_from_embeddings()` generates sine waves (`0.3 * torch.sin(...)`) when inference fails.
- **Consequence:** Zero pronunciation learning occurs for CosyVoice, and synthetic tone placeholders are returned as speech.
- **Fix Applied:** Eliminate all dummy losses and sine-wave fallbacks. Introduce `PronunciationConditioningAdapter` and candidate-selection optimization for non-differentiable speech-token generation pipelines.

### 3. Unsafe Reference Transcript Mutation in F5-TTS
- **File Path:** `flowedit/backbone/f5tts_wrapper.py`
- **Functions:** `synthesize_from_embeddings()`, `synthesize_direct()`
- **Problem:** Logic deletes reference words from `ref_text` if they appear in `gen_text`: `[w for w in ref_words if w not in gen_words]`.
- **Consequence:** Destroys alignment between reference audio and reference transcript, causing boundary confusion and audio bleeding.
- **Fix Applied:** Remove transcript stripping. Validate reference transcripts using Whisper/WhisperX CER/WER alignment confidence and preserve exact transcript-audio pairing.

### 4. Flawed Embedding Alignment Assumption in F5-TTS
- **File Path:** `flowedit/backbone/f5tts_wrapper.py`
- **Function:** `synthesize_from_embeddings()` (Hook logic)
- **Problem:** Assumes target embeddings occur at `start_pos = T_mel - L_gen`.
- **Consequence:** Misaligns phonetic edits with actual target token positions, leading to garbled audio output.
- **Fix Applied:** Build an exact sequence token-to-mel frame mapping taking into account prompt padding, sequence lengths, and duration expansion.

### 5. Incorrect CosyVoice Inference Mode Selection & Zero Speaker Vector
- **File Path:** `flowedit/backbone/cosyvoice_wrapper.py`
- **Functions:** `get_speaker_embedding()`, `synthesize_from_embeddings()`
- **Problem:** Returns dummy zero speaker vectors (`torch.zeros(1, 192)`). Does not select `inference_zero_shot` vs `inference_cross_lingual` based on whether `prompt_language == target_language`.
- **Consequence:** Distorts voice identity and applies wrong conditioning routines (e.g. using zero-shot mode for Chinese prompt + English target).
- **Fix Applied:** Use `CosyVoice2` class and dynamically route `prompt_language == target_language` to `inference_zero_shot` and mixed languages to `inference_cross_lingual`.

### 6. Leaky Cross-Backbone Pronunciation Memory
- **File Path:** `flowedit/memory/hopfield_memory.py`
- **Functions:** `store()`, `retrieve()`
- **Problem:** Stores raw latent vectors without recording backbone type, model version, tokenizer, speaker ID, or language.
- **Consequence:** A correction learned on F5-TTS is incorrectly retrieved and applied to CosyVoice or XTTS, causing severe embedding distortion.
- **Fix Applied:** Redesign memory with `PronunciationCorrection` dataclass. Strict filtering ensures corrections are never shared across incompatible backbones or model versions.

### 7. Missing Dynamic Duration Planning and Naive Speed Factors
- **File Path:** `flowedit/backbone/f5tts_wrapper.py`
- **Problem:** Relies on hardcoded or global speed adjustments (e.g., `speed=0.88`), leading to rushed speech or cut-off sentence endings.
- **Consequence:** Unnatural pacing, clipped final words.
- **Fix Applied:** Implement `flowedit/text/duration_planner.py` deriving cadence from prompt voiced duration, syllable/phoneme counts, punctuation pauses, and sentence padding.

### 8. Unauthorized API Endpoints & Unprotected Text Splitting
- **File Path:** `flowedit/api/main.py`
- **Endpoints:** `/api/synthesize_raw` added violating backward compatibility constraints. Text splitting in chunkers divides named entities.
- **Consequence:** Inconsistent API contract and broken entity pronunciations (e.g. splitting "Mrunmayee Sakharwade" across chunks).
- **Fix Applied:** Remove `/api/synthesize_raw`. Preserve existing `/api/synthesize` and `/api/correct` endpoints. Enforce entity-safe chunking in text processor.

---
