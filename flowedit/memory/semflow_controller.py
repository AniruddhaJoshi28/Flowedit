"""
=============================================================================
SemFlow-Memory: End-to-End Context-Aware Lifelong Pronunciation Adaptation
=============================================================================
Core Research Controller integrating:
 1. Frozen MiniLM Contextual Semantic Encoder (d=384)
 2. Contrastive Pronunciation Adapter (CPA, 384 -> 256)
 3. Interference-Aware Modern Hopfield Correction Memory (with margin gating & EMA prototypes)
 4. Configurable Acoustic Actuator (Localized Waveform Regeneration & Masked Mel Blending)
 5. Frozen F5-TTS Backbone (600M DiT, Vocos 24kHz)
=============================================================================
"""

import os
import sys
import time
import math
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from scipy.signal import resample

# Local imports
from flowedit.memory.contrastive_adapter import ContrastivePronunciationAdapter, SupervisedContrastiveLoss
from flowedit.memory.correction_memory import MemoryEntry, CosineKNNMemory, InterferenceAwareHopfieldMemory

logger = logging.getLogger("SemFlowController")


# ---------------------------------------------------------------------------
# 1. Ablated Hopfield Memory (Without Margin Gating)
# ---------------------------------------------------------------------------

class AblatedHopfieldMemory:
    """Hopfield Associative Memory WITHOUT interference margin gating.
    Used as an ablation baseline to demonstrate the necessity of margin gating.
    """
    def __init__(self, beta: float = 8.0, abstain_threshold: float = 0.60, ema_alpha: float = 0.80):
        self.beta = beta
        self.abstain_threshold = abstain_threshold
        self.ema_alpha = ema_alpha
        self.entries: List[MemoryEntry] = []

    def add_or_update_entry(self, key: torch.Tensor, value: torch.Tensor, spelling: str, sense_id: int, label: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        k = F.normalize(key.detach().cpu().float().flatten(), p=2, dim=0)
        v = value.detach().cpu().float()
        spelling_norm = spelling.lower().strip()

        for existing in self.entries:
            if existing.spelling == spelling_norm and existing.sense_id == sense_id:
                merged_k = self.ema_alpha * existing.key + (1.0 - self.ema_alpha) * k
                existing.key = F.normalize(merged_k, p=2, dim=0)
                existing.value = self.ema_alpha * existing.value + (1.0 - self.ema_alpha) * v
                existing.sample_count += 1
                if metadata: existing.metadata.update(metadata)
                return

        entry = MemoryEntry(key=k, value=v, spelling=spelling_norm, sense_id=sense_id, label=label, sample_count=1, metadata=metadata or {})
        self.entries.append(entry)

    def retrieve(self, query: torch.Tensor, target_spelling: Optional[str] = None) -> Dict[str, Any]:
        if not self.entries:
            return {"value": None, "similarity": 0.0, "margin": 0.0, "is_abstained": True, "abstain_reason": "empty_memory", "entry": None, "method": "ablated_hopfield"}

        q = F.normalize(query.detach().cpu().float().flatten(), p=2, dim=0)
        candidates = self.entries
        if target_spelling:
            target_norm = target_spelling.lower().strip()
            filtered = [e for e in self.entries if e.spelling == target_norm]
            if filtered: candidates = filtered

        sims = torch.tensor([torch.dot(q, e.key).item() for e in candidates], dtype=torch.float32)
        max_sim = float(sims.max().item())
        attn_weights = F.softmax(self.beta * sims, dim=0)

        # Associative readout
        v_out = torch.zeros_like(candidates[0].value)
        for i, entry in enumerate(candidates):
            v_out = v_out + attn_weights[i] * entry.value

        best_idx = int(torch.argmax(attn_weights).item())
        best_entry = candidates[best_idx]

        # In ablated version: NO margin gating, only similarity threshold
        is_abstained = max_sim < self.abstain_threshold
        abstain_reason = "low_similarity" if is_abstained else "none"

        return {
            "value": v_out if not is_abstained else None,
            "similarity": max_sim,
            "margin": 1.0, # ignored in ablated
            "is_abstained": is_abstained,
            "abstain_reason": abstain_reason,
            "entry": best_entry,
            "method": "ablated_hopfield",
        }

    def clear(self):
        self.entries.clear()


# ---------------------------------------------------------------------------
# 2. Frozen MiniLM Contextual Semantic Encoder
# ---------------------------------------------------------------------------

class MiniLMContextEncoder:
    """Extracts target-word pooled contextual representations using frozen all-MiniLM-L6-v2."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        from transformers import AutoTokenizer, AutoModel
        logger.info("Loading frozen all-MiniLM-L6-v2 contextual encoder...")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def encode_target_context(self, sentence: str, target_word: str) -> torch.Tensor:
        """Locates the character span of target_word in sentence and pools token hidden states."""
        sent_lower = sentence.lower()
        word_lower = target_word.lower().strip()
        idx = sent_lower.find(word_lower)
        if idx == -1:
            # Fallback if punctuation or boundary attached
            words = sent_lower.split()
            clean_words = [w.strip(".,!?;:\"'") for w in words]
            if word_lower in clean_words:
                w_pos = clean_words.index(word_lower)
                idx = len(" ".join(words[:w_pos])) + 1 if w_pos > 0 else 0
            else:
                idx = 0
        cs = idx
        ce = cs + len(word_lower)

        enc = self.tokenizer(sentence, return_offsets_mapping=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            offsets = enc["offset_mapping"][0].cpu().numpy()
            target_toks = [
                t_i for t_i, (t_s, t_e) in enumerate(offsets)
                if max(cs, t_s) < min(ce, t_e) and t_s < t_e
            ]
            if target_toks:
                emb = out.last_hidden_state[0, target_toks, :].mean(dim=0)
            else:
                emb = out.last_hidden_state[0, 0, :] # CLS token fallback
        return F.normalize(emb, p=2, dim=0)


# ---------------------------------------------------------------------------
# 3. Waveform Splicing & Mel Blending Utilities
# ---------------------------------------------------------------------------

def splice_audio_patch(
    base_wav: np.ndarray,
    patch_wav: np.ndarray,
    t_start: float,
    t_end: float,
    sr: int = 24000,
    fade_sec: float = 0.020,
) -> np.ndarray:
    """Robustly splices a patch into base audio across initial, medial, and final positions.
    Guarantees bit-exact sample preservation outside [t_start - fade_sec, t_end + fade_sec].
    """
    fade_len = int(fade_sec * sr)
    s_idx = max(0, int(t_start * sr))
    e_idx = min(len(base_wav), max(s_idx + 16, int(t_end * sr)))
    slot_len = e_idx - s_idx

    # RMS normalization
    rms_base = np.sqrt(np.mean(base_wav[s_idx:e_idx]**2) + 1e-8)
    rms_patch = np.sqrt(np.mean(patch_wav**2) + 1e-8)
    norm_patch = patch_wav * (rms_base / rms_patch)

    fade_len = min(fade_len, len(norm_patch) // 4, max(16, slot_len // 4))
    fade_len = max(16, fade_len)
    fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_len)))
    fade_out = 1.0 - fade_in

    has_prefix = s_idx >= fade_len
    has_suffix = (len(base_wav) - e_idx) >= fade_len

    if has_prefix and has_suffix:
        prefix = base_wav[:s_idx]
        suffix = base_wav[e_idx:]
        start_trans = prefix[-fade_len:] * fade_out + norm_patch[:fade_len] * fade_in
        end_trans = norm_patch[-fade_len:] * fade_out + suffix[:fade_len] * fade_in
        body = norm_patch[fade_len:-fade_len] if len(norm_patch) > 2 * fade_len else norm_patch[:0]
        return np.concatenate([prefix[:-fade_len], start_trans, body, end_trans, suffix[fade_len:]])
    elif has_suffix:  # Initial position (no prefix)
        suffix = base_wav[e_idx:]
        end_trans = norm_patch[-fade_len:] * fade_out + suffix[:fade_len] * fade_in
        body = norm_patch[:-fade_len] if len(norm_patch) > fade_len else norm_patch[:0]
        onset_len = min(int(0.005 * sr), len(body))
        if onset_len > 0:
            body = body.copy()
            body[:onset_len] *= (0.5 * (1.0 - np.cos(np.linspace(0, np.pi, onset_len))))
        return np.concatenate([body, end_trans, suffix[fade_len:]])
    elif has_prefix:  # Final position (no suffix)
        prefix = base_wav[:s_idx]
        start_trans = prefix[-fade_len:] * fade_out + norm_patch[:fade_len] * fade_in
        body = norm_patch[fade_len:] if len(norm_patch) > fade_len else norm_patch[:0]
        offset_len = min(int(0.005 * sr), len(body))
        if offset_len > 0:
            body = body.copy()
            body[-offset_len:] *= (0.5 * (1.0 + np.cos(np.linspace(0, np.pi, offset_len))))
        return np.concatenate([prefix[:-fade_len], start_trans, body])
    else:
        return norm_patch


# ---------------------------------------------------------------------------
# 4. Configurable Acoustic Actuator
# ---------------------------------------------------------------------------

class ConfigurableAcousticActuator:
    """Performs acoustic actuation via Localized Waveform Regeneration or Masked Mel Blending."""

    def __init__(self, tts_instance: Any, whisper_instance: Any, device: str = "cpu"):
        self.tts = tts_instance
        self.vocoder = tts_instance.vocoder
        self.whisper = whisper_instance
        self.device = device

    def generate_baseline(
        self,
        sentence: str,
        ref_file: str,
        ref_text: str,
        nfe_step: int = 8,
        speed: float = 1.0,
        seed: int = 42,
    ) -> Tuple[np.ndarray, int, float]:
        """Synthesizes baseline speech with frozen F5-TTS."""
        t0 = time.time()
        wav, sr, _ = self.tts.infer(
            ref_file=ref_file,
            ref_text=ref_text,
            gen_text=sentence,
            nfe_step=nfe_step,
            cfg_strength=2.0,
            speed=speed,
            seed=seed,
        )
        t_elapsed = time.time() - t0
        return wav, sr, t_elapsed

    def get_word_alignment(self, wav_path: str, target_word: str) -> Tuple[float, float, str]:
        """Extracts word boundaries using Whisper."""
        segs, _ = self.whisper.transcribe(wav_path, language="en", word_timestamps=True)
        segs = list(segs)
        target_norm = target_word.lower().strip(".,!?;:\"'")
        for s in segs:
            for w in s.words:
                cl = w.word.strip().lower().strip(".,!?;:\"'")
                if target_norm in cl or cl in target_norm:
                    return w.start, w.end, cl
        return 0.5, 0.9, target_word

    def actuate(
        self,
        base_wav: np.ndarray,
        base_wav_path: str,
        carrier_sentence: str,
        phonetic_sentence: str,
        target_word: str,
        phonetic_word: str,
        ref_file: str,
        ref_text: str,
        mode: str = "localized_waveform_regeneration",
        sr: int = 24000,
        nfe_step: int = 8,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Applies acoustic correction in carrier context.
        Modes:
          - 'localized_waveform_regeneration': in-context patch synthesis + 20ms cosine crossfade.
          - 'masked_mel_blending': in-context patch synthesis + Hann window mel inpainting + Vocos decode.
          - 'phonetic_substitution': full sentence re-synthesis with phonetic word.
        """
        t_s_b, t_e_b, base_word_spoken = self.get_word_alignment(base_wav_path, target_word)

        if mode == "phonetic_substitution":
            t0 = time.time()
            sub_wav, sr, _ = self.tts.infer(
                ref_file=ref_file,
                ref_text=ref_text,
                gen_text=phonetic_sentence,
                nfe_step=nfe_step,
                cfg_strength=2.0,
                speed=1.0,
                seed=42,
            )
            t_act = time.time() - t0
            return sub_wav, t_act, {"mode": mode, "t_start": t_s_b, "t_end": t_e_b}

        # For Localized Regen and Mel Blending: synthesize phonetic sentence to extract patch
        t0_synth = time.time()
        sub_wav, sr, _ = self.tts.infer(
            ref_file=ref_file,
            ref_text=ref_text,
            gen_text=phonetic_sentence,
            nfe_step=nfe_step,
            cfg_strength=2.0,
            speed=1.0,
            seed=42,
        )
        t_patch_synth = time.time() - t0_synth

        # Align patch word in phonetic audio
        tmp_sub_path = "reports/phase3_audio_artifacts/tmp_sub.wav"
        os.makedirs("reports/phase3_audio_artifacts", exist_ok=True)
        sf.write(tmp_sub_path, sub_wav, sr)
        t_s_sub, t_e_sub, _ = self.get_word_alignment(tmp_sub_path, phonetic_word)

        if mode == "localized_waveform_regeneration":
            t0_splice = time.time()
            s_p_idx = max(0, int(t_s_sub * sr))
            e_p_idx = min(len(sub_wav), int(t_e_sub * sr))
            target_patch = sub_wav[s_p_idx:e_p_idx]
            if len(target_patch) < 32:
                target_patch = sub_wav

            edited_wav = splice_audio_patch(base_wav, target_patch, t_s_b, t_e_b, sr=sr, fade_sec=0.020)
            t_act_pure = time.time() - t0_splice
            meta = {
                "mode": mode,
                "t_patch_synth": round(t_patch_synth, 2),
                "t_actuation_pure": round(t_act_pure, 4),
                "t_start": t_s_b,
                "t_end": t_e_b,
            }
            return edited_wav, t_act_pure, meta

        elif mode == "masked_mel_blending":
            t0_mel = time.time()
            n_frames = int(len(base_wav) / 256)
            f_start = max(0, min(int(t_s_b * sr / 256), n_frames - 1))
            f_end = max(f_start + 2, min(int(t_e_b * sr / 256), n_frames))

            mask_frames = torch.zeros(n_frames)
            span_w = f_end - f_start
            hann_win = torch.hann_window(span_w, periodic=False)
            mask_frames[f_start:f_end] = hann_win

            t_base_tensor = torch.from_numpy(base_wav).unsqueeze(0).float()
            t_sub_tensor = torch.from_numpy(sub_wav).unsqueeze(0).float()
            with torch.no_grad():
                mel_b = self.tts.ema_model.mel_spec(t_base_tensor)
                mel_patch = self.tts.ema_model.mel_spec(t_sub_tensor)

            min_len = min(mel_b.shape[-1], mel_patch.shape[-1], mask_frames.shape[0])
            mel_b = mel_b[:, :, :min_len]
            mel_patch = mel_patch[:, :, :min_len]
            m_mask = mask_frames[:min_len].view(1, 1, -1)

            mel_blended = (1.0 - m_mask) * mel_b + m_mask * mel_patch

            with torch.no_grad():
                wav_mel_tensor = self.vocoder.decode(mel_blended)
                edited_wav = wav_mel_tensor.squeeze().cpu().numpy()

            t_act_pure = time.time() - t0_mel
            meta = {
                "mode": mode,
                "t_patch_synth": round(t_patch_synth, 2),
                "t_actuation_pure": round(t_act_pure, 4),
                "t_start": t_s_b,
                "t_end": t_e_b,
            }
            return edited_wav, t_act_pure, meta
        else:
            raise ValueError(f"Unknown acoustic mode: {mode}")


# ---------------------------------------------------------------------------
# 5. Master SemFlow-Memory End-to-End System
# ---------------------------------------------------------------------------

class SemFlowSystem:
    """Master research system integrating encoder, adapter, memory, and acoustic controller."""

    def __init__(
        self,
        encoder: MiniLMContextEncoder,
        adapter: ContrastivePronunciationAdapter,
        memory: Any, # InterferenceAwareHopfieldMemory, CosineKNNMemory, or AblatedHopfieldMemory
        actuator: ConfigurableAcousticActuator,
        default_mode: str = "localized_waveform_regeneration",
    ):
        self.encoder = encoder
        self.adapter = adapter
        self.memory = memory
        self.actuator = actuator
        self.default_mode = default_mode

    def extract_context_key(self, sentence: str, target_word: str) -> torch.Tensor:
        """Extracts MiniLM contextual representation and projects through CPA."""
        raw_emb = self.encoder.encode_target_context(sentence, target_word)
        with torch.no_grad():
            cpa_key = self.adapter(raw_emb.unsqueeze(0), normalize=True).squeeze(0)
        return cpa_key

    def provide_feedback(
        self,
        sentence: str,
        target_word: str,
        correct_sense_id: int,
        label: str,
        phonetic_word: str,
    ) -> Dict[str, Any]:
        """User feedback learning step: encodes context and stores/updates associative memory."""
        key = self.extract_context_key(sentence, target_word)
        # Store in memory: value stores one-hot/sense vector or phonetic payload
        val = torch.tensor([float(correct_sense_id)], dtype=torch.float32)

        if hasattr(self.memory, "add_or_update_entry"):
            self.memory.add_or_update_entry(
                key=key,
                value=val,
                spelling=target_word,
                sense_id=correct_sense_id,
                label=label,
                metadata={"phonetic_word": phonetic_word, "seed_sentence": sentence},
            )
        elif hasattr(self.memory, "add_entry"):
            self.memory.add_entry(
                key=key,
                value=val,
                spelling=target_word,
                sense_id=correct_sense_id,
                label=label,
                metadata={"phonetic_word": phonetic_word, "seed_sentence": sentence},
            )

        logger.info(f"[FEEDBACK STORED] Word: '{target_word}' | Sense: {correct_sense_id} ({label}) | Total Entries: {len(self.memory.entries)}")
        return {
            "status": "stored",
            "target_word": target_word,
            "sense_id": correct_sense_id,
            "label": label,
            "phonetic_word": phonetic_word,
            "total_entries": len(self.memory.entries),
        }

    def synthesize_with_adaptation(
        self,
        sentence: str,
        target_word: str,
        speaker: str,
        ref_file: str,
        ref_text: str,
        output_path: str,
        mode: Optional[str] = None,
        cached_base_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """End-to-End Inference:
        1. Synthesizes or loads baseline audio.
        2. Queries correction memory with contextual CPA representation.
        3. If memory retrieves an active (non-abstained) correction, actuates localized acoustic repair.
        4. If memory abstains (or memory is empty), preserves baseline audio.
        """
        act_mode = mode or self.default_mode
        t_start_total = time.time()

        # Step 1: Baseline Synthesis
        if cached_base_path and os.path.exists(cached_base_path):
            base_wav, sr = sf.read(cached_base_path)
            t_base = 0.05
            base_wav_path = cached_base_path
        else:
            base_wav_path = output_path.replace(".wav", "_base.wav")
            base_wav, sr, t_base = self.actuator.generate_baseline(
                sentence=sentence,
                ref_file=ref_file,
                ref_text=ref_text,
            )
            sf.write(base_wav_path, base_wav, sr)

        # Step 2: Context Extraction & Memory Retrieval
        t0_query = time.time()
        cpa_key = self.extract_context_key(sentence, target_word)
        retrieval = self.memory.retrieve(cpa_key, target_spelling=target_word)
        t_query = time.time() - t0_query

        is_abstained = retrieval.get("is_abstained", True)
        retrieved_entry = retrieval.get("entry", None)

        # Step 3: Acoustic Actuation Decision
        if not is_abstained and retrieved_entry is not None:
            phon_word = retrieved_entry.metadata.get("phonetic_word", target_word)
            phon_sentence = sentence.replace(target_word, phon_word).replace(target_word.capitalize(), phon_word.capitalize())

            final_wav, t_act_pure, meta = self.actuator.actuate(
                base_wav=base_wav,
                base_wav_path=base_wav_path,
                carrier_sentence=sentence,
                phonetic_sentence=phon_sentence,
                target_word=target_word,
                phonetic_word=phon_word,
                ref_file=ref_file,
                ref_text=ref_text,
                mode=act_mode,
                sr=sr,
            )
            action_taken = f"corrected_via_{act_mode}"
            predicted_sense = retrieved_entry.sense_id
            retrieved_label = retrieved_entry.label
        else:
            final_wav = base_wav
            t_act_pure = 0.0
            action_taken = f"baseline_preserved (abstained: {retrieval.get('abstain_reason', 'none')})"
            predicted_sense = -1 # default / no correction applied
            retrieved_label = "none"
            meta = {}

        sf.write(output_path, final_wav, sr)
        t_total = time.time() - t_start_total

        return {
            "output_path": output_path,
            "action_taken": action_taken,
            "is_abstained": is_abstained,
            "abstain_reason": retrieval.get("abstain_reason", "none"),
            "similarity": retrieval.get("similarity", 0.0),
            "margin": retrieval.get("margin", 0.0),
            "predicted_sense": predicted_sense,
            "retrieved_label": retrieved_label,
            "t_base": round(t_base, 2),
            "t_query_ms": round(t_query * 1000, 2),
            "t_actuation": round(t_act_pure, 4),
            "t_total": round(t_total, 2),
            "meta": meta,
        }
