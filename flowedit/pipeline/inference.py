"""
FlowEdit Inference Pipeline — Synthesis with Pronunciation Corrections (arXiv:2606.20518).

Paper Section 3.2 (Stage 3 & Inference):
    "During inference, text embeddings c are refined using the associative memory:
     c_hat = c + σ(max_j (β Q K_j^T) - τ) ⊙ Mem(Q)
     where Mem(Q) = softmax(β Q K^T) V, β = 1/√d."
"""

import time
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import torch
import soundfile as sf

from flowedit.config import FlowEditConfig
from flowedit.backbone import create_backbone, F5TTSBackbone
from flowedit.memory.hopfield_memory import HopfieldMemory
from flowedit.refiner.hopfield_refiner import HopfieldRefiner
from flowedit.utils.indic_phonetics import normalize_indic_phonetics

logger = logging.getLogger(__name__)


class FlowEditInference:
    """Production inference pipeline with continuous Hopfield associative pronunciation memory."""

    def __init__(self, config: Optional[FlowEditConfig] = None):
        self.config = config or FlowEditConfig()
        self.backbone: Optional[F5TTSBackbone] = None
        self.memory: Optional[HopfieldMemory] = None
        self.refiner: Optional[HopfieldRefiner] = None
        self._is_ready = False

    def load(
        self,
        backbone: Optional[F5TTSBackbone] = None,
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
        speaker_wav: str,
        language: str = "en",
        user_ref_text: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synthesize speech with automatic Hopfield memory pronunciation retrieval.

        Args:
            text: Input carrier sentence
            speaker_wav: Speaker voice audio
            language: Language code
            user_ref_text: Optional speaker text
            output_path: Optional output .wav path

        Returns:
            Dict containing waveform, sample_rate, is_modified, and diagnostics
        """
        self._ensure_ready()
        t0 = time.time()

        text = normalize_indic_phonetics(text)
        speaker_conditioning = self.backbone.get_speaker_embedding(
            speaker_wav, language=language, ref_text=user_ref_text
        )

        # Refine text embeddings and synthesize
        refine_res = self.refiner(
            backbone=self.backbone,
            text=text,
            speaker_conditioning=speaker_conditioning,
            language=language,
            user_ref_text=user_ref_text,
        )

        waveform = refine_res.waveform
        sr = refine_res.sample_rate

        if output_path:
            sf.write(output_path, waveform.squeeze().cpu().numpy(), sr)
            logger.info(f"Saved synthesized audio to {output_path}")

        elapsed_ms = (time.time() - t0) * 1000.0

        return {
            "waveform": waveform,
            "sample_rate": sr,
            "is_modified": refine_res.is_modified,
            "inference_time_ms": elapsed_ms,
            "diagnostics": refine_res.diagnostics,
        }
