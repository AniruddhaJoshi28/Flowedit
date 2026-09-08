"""
FlowEdit Inference Pipeline — Synthesis with Pronunciation Corrections (arXiv:2606.20518).

Paper Section 3.2 (Stage 3 & Inference):
    "During inference, text embeddings c are refined using the associative memory:
     c_hat = c + σ(max_j (β Q K_j^T) - τ) ⊙ Mem(Q)
     where Mem(Q) = softmax(β Q K^T) V, β = 1/√d."
"""

import os
import time
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import torch
import numpy as np
try:
    import soundfile as sf
except ImportError:
    sf = None

from flowedit.config import FlowEditConfig
from flowedit.backbone import create_backbone, TTSBackbone
from flowedit.memory.hopfield_memory import HopfieldMemory
from flowedit.refiner.hopfield_refiner import HopfieldRefiner
from flowedit.utils.indic_phonetics import normalize_indic_phonetics

logger = logging.getLogger(__name__)


class FlowEditInference:
    """Production inference pipeline with continuous Hopfield associative pronunciation memory."""

    def __init__(self, config: Optional[FlowEditConfig] = None):
        self.config = config or FlowEditConfig()
        self.backbone: Optional[TTSBackbone] = None
        self.memory: Optional[HopfieldMemory] = None
        self.refiner: Optional[HopfieldRefiner] = None
        self._is_ready = False

    def load(
        self,
        backbone: Optional[TTSBackbone] = None,
        memory: Optional[HopfieldMemory] = None,
    ) -> None:
        """Initialize inference pipeline."""
        logger.info("Initializing FlowEdit inference pipeline...")

        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = create_backbone(self.config.backbone)
            self.backbone.load_model()

        embed_dim = self.backbone.embedding_dim

        if memory is not None:
            self.memory = memory
        else:
            self.memory = HopfieldMemory(self.config.memory, embedding_dim=embed_dim)

        self.refiner = HopfieldRefiner(memory=self.memory, config=self.config.memory)
        self._is_ready = True
        logger.info(f"✓ Inference pipeline ready. Memory has {self.memory.num_entries} corrections.")

    def _ensure_ready(self) -> None:
        if not self._is_ready:
            self.load()

    def synthesize(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
        user_ref_text: Optional[str] = None,
        output_path: Optional[str] = None,
        speaker_name: Optional[str] = "female",
        **kwargs,
    ) -> Dict[str, Any]:
        """Synthesize speech with automatic Hopfield memory pronunciation retrieval.

        Args:
            text: Input carrier sentence
            speaker_wav: Speaker voice audio (optional, defaults to preset voice)
            language: Language code
            user_ref_text: Optional speaker text
            output_path: Optional output .wav path
            speaker_name: Preset voice name if speaker_wav not provided ('female'/'blessing' or 'male'/'michael')

        Returns:
            Dict containing waveform, sample_rate, is_modified, and diagnostics
        """
        self._ensure_ready()
        t0 = time.time()
        text = normalize_indic_phonetics(text)

        # Ensure speaker_wav is valid; if not provided or missing, resolve to hardcoded preset voice
        if not speaker_wav or not os.path.isfile(speaker_wav):
            from flowedit.api.main import get_hardcoded_voice_path
            target = "michael" if (speaker_name or "").strip().lower() in ("male", "man", "michael") else "blessing"
            resolved = get_hardcoded_voice_path(target)
            if resolved and os.path.isfile(resolved):
                speaker_wav = resolved
                logger.info(f"Using hardcoded {target} voice for synthesis: {speaker_wav}")
            else:
                raise FileNotFoundError(f"Speaker audio not provided and hardcoded voice for '{target}' not found on disk.")

        # Autonomous Memory Check 1: S3 Spelling Store (deterministic phonetic respelling via shared HomographContextResolver)
        from flowedit.memory.s3_storage import s3_spelling_store
        text, spelling_applied = s3_spelling_store.apply_corrections_to_text(text)
        if spelling_applied:
            logger.info(f"[Inference] Autonomous S3 spelling correction applied: {spelling_applied}")

        speaker_conditioning = self.backbone.get_speaker_embedding(
            speaker_wav, language=language, ref_text=user_ref_text
        )

        # Autonomous Memory Check 2: Modern Hopfield Continuous Memory (continuous latent delta retrieval)
        refine_res = self.refiner(
            backbone=self.backbone,
            text=text,
            speaker_conditioning=speaker_conditioning,
            language=language,
            user_ref_text=user_ref_text,
            **kwargs,
        )

        waveform = refine_res.waveform
        sr = refine_res.sample_rate

        if output_path:
            wav_np = waveform.squeeze().cpu().numpy()
            if sf is not None:
                sf.write(output_path, wav_np, sr)
            else:
                import wave
                int16_d = (np.clip(wav_np, -1.0, 1.0) * 32767.0).astype(np.int16)
                with wave.open(output_path, "wb") as wf:
                    wf.setnchannels(1 if wav_np.ndim == 1 else wav_np.shape[1])
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(int16_d.tobytes())
            logger.info(f"Saved synthesized audio to {output_path}")

        elapsed_ms = (time.time() - t0) * 1000.0

        diag = dict(refine_res.diagnostics)
        diag["spelling_applied"] = spelling_applied

        return {
            "waveform": waveform,
            "sample_rate": sr,
            "is_modified": refine_res.is_modified or bool(spelling_applied),
            "spelling_applied": spelling_applied,
            "hopfield_active": refine_res.is_modified,
            "inference_time_ms": elapsed_ms,
            "diagnostics": diag,
        }
