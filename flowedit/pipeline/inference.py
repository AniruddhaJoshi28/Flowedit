"""
FlowEdit Inference Pipeline — Synthesis with Pronunciation Corrections.

Paper Figure 1 (Left): "Inference pipeline. Input text is encoded by a
frozen Text Encoder and refined by the Hopfield Refiner, which retrieves
stored corrections via soft attention. The refined embeddings are decoded
by the frozen DiT."

Flow at inference:
    text → TextEncoder → c → HopfieldRefiner → ĉ → FrozenDecoder → audio

The HopfieldRefiner applies corrections only when the similarity gate
activates (cosine similarity exceeds threshold τ). For general text
without matching corrections, the gate outputs ~0 and the decoder
receives EXACTLY the same embeddings as the base model.
"""

import torch
import logging
import time
from typing import Optional, Dict, Tuple
from pathlib import Path

from flowedit.config import FlowEditConfig
from flowedit.backbone import create_backbone, F5TTSBackbone
from flowedit.memory.hopfield_memory import HopfieldMemory
from flowedit.refiner.hopfield_refiner import HopfieldRefiner
from flowedit.utils.audio import AudioProcessor

logger = logging.getLogger(__name__)


class FlowEditInference:
    """Production inference pipeline with pronunciation correction.

    This is the deployment-time interface. After corrections have been
    learned via CorrectionLoop, this class synthesizes speech with those
    corrections automatically applied.

    Key properties (from paper):
    - Zero forgetting: general speech quality is IDENTICAL to base model
    - Fuzzy matching: "Linux" correction partially applies to "Linux's"
    - Speaker-agnostic: corrections work across all voices
    - Low latency: retrieval overhead < 35ms for M ≤ 500 corrections
    """

    def __init__(self, config: Optional[FlowEditConfig] = None):
        """Initialize inference pipeline.

        Args:
            config: FlowEdit configuration
        """
        self.config = config or FlowEditConfig()
        self.backbone: Optional[F5TTSBackbone] = None
        self.memory: Optional[HopfieldMemory] = None
        self.refiner: Optional[HopfieldRefiner] = None
        self.audio_processor = AudioProcessor(self.config.audio)
        self._is_ready = False

    def load(
        self,
        memory_path: Optional[str] = None,
        backbone: Optional[F5TTSBackbone] = None,
        memory: Optional[HopfieldMemory] = None,
    ) -> None:
        """Load models and memory for inference.

        Args:
            memory_path: Path to saved Hopfield memory file
            backbone: Optional pre-loaded backbone (avoids reloading)
            memory: Optional pre-initialized memory (avoids reloading)
        """
        logger.info("Loading FlowEdit inference pipeline...")

        # Load backbone
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = create_backbone(self.config.backbone)
            self.backbone.load_model()

        embed_dim = self.backbone.embedding_dim

        # Load memory
        if memory is not None:
            self.memory = memory
        else:
            self.memory = HopfieldMemory(dim=embed_dim, config=self.config.memory)
            if memory_path and Path(memory_path).exists():
                self.memory.load(memory_path)

        # Initialize refiner
        self.refiner = HopfieldRefiner(
            memory=self.memory,
            config=self.config.memory,
        )
        self.refiner = self.refiner.to(self.backbone.device)

        self._is_ready = True

        logger.info(
            f"Inference pipeline ready. "
            f"Memory: {self.memory.size} corrections loaded."
        )

    def synthesize(
        self,
        text: str,
        speaker_wav: str,
        language: str = "en",
        output_path: Optional[str] = None,
        return_gate_info: bool = False,
    ) -> Dict:
        """Synthesize speech with automatic pronunciation corrections.

        Pipeline:
            text → encode → refine (Hopfield) → decode (F5-TTS) → audio

        Args:
            text: Input text to synthesize
            speaker_wav: Speaker reference audio for voice cloning
            language: Language code
            output_path: Optional path to save output audio
            return_gate_info: If True, include gate analysis in result

        Returns:
            Dict with:
            - 'waveform': Audio tensor [1, T]
            - 'sample_rate': Sample rate
            - 'corrections_applied': Number of tokens with active corrections
            - 'gate_info': Optional gate analysis (if return_gate_info=True)
            - 'inference_time_ms': Total inference time in milliseconds
        """
        self._ensure_ready()

        start_time = time.time()

        # Apply Indic phonetic normalizer (e.g. Mrunmayee -> Mroonmayee)
        from flowedit.utils.indic_phonetics import normalize_indic_phonetics
        text = normalize_indic_phonetics(text)

        # Step 1: Encode text → c
        with torch.no_grad():
            text_embeddings = self.backbone.encode_text(text, language)
            # text_embeddings: [1, seq_len, dim]

        # Step 2: Refine via HopfieldRefiner → ĉ
        # This is where corrections are applied (or passed through)
        with torch.no_grad():
            def token_locator(prefix_str):
                # Returns the number of tokens in the prefix string
                tokens = self.backbone.get_token_ids(prefix_str, language)
                return tokens.shape[1]
                
            refined_embeddings, gate_values = self.refiner(
                text_embeddings, 
                text=text,
                token_locator=token_locator,
            )

        # Count active corrections
        corrections_applied = (gate_values > 0.5).sum().item()

        if corrections_applied > 0:
            logger.info(
                f"Applied corrections to {corrections_applied} tokens "
                f"(max gate: {gate_values.max().item():.3f})"
            )
        else:
            logger.debug("No corrections applied (all gates < 0.5)")

        # Step 3: Decode with F5-TTS → audio
        speaker_conditioning = self.backbone.get_speaker_embedding(speaker_wav)

        waveform, sr = self.backbone.synthesize_from_embeddings(
            text_embeddings=refined_embeddings,
            speaker_conditioning=speaker_conditioning,
            text=text,
            language=language,
        )

        inference_time_ms = (time.time() - start_time) * 1000

        # Save if output path specified
        if output_path:
            self.audio_processor.save_audio(
                waveform,
                output_path,
                sr,
            )
            logger.info(f"Audio saved to {output_path}")

        result = {
            "waveform": waveform,
            "sample_rate": self.config.audio.sample_rate,
            "corrections_applied": int(corrections_applied),
            "inference_time_ms": inference_time_ms,
        }

        # Optional gate analysis
        if return_gate_info:
            result["gate_info"] = self.refiner.get_gate_analysis(
                text_embeddings.squeeze(0)
            )

        return result

    def synthesize_baseline(
        self,
        text: str,
        speaker_wav: str,
        language: str = "en",
        output_path: Optional[str] = None,
    ) -> Dict:
        """Synthesize WITHOUT corrections (baseline comparison).

        Useful for A/B testing and verifying zero forgetting.

        Args:
            text: Input text
            speaker_wav: Speaker reference
            language: Language code
            output_path: Optional output path

        Returns:
            Dict with waveform and metadata
        """
        self._ensure_ready()

        start_time = time.time()

        waveform = self.backbone.synthesize_direct(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
        )

        inference_time_ms = (time.time() - start_time) * 1000

        if output_path:
            self.audio_processor.save_audio(
                waveform, output_path, self.config.audio.sample_rate
            )

        return {
            "waveform": waveform,
            "sample_rate": self.config.audio.sample_rate,
            "corrections_applied": 0,
            "inference_time_ms": inference_time_ms,
        }

    def compare(
        self,
        text: str,
        speaker_wav: str,
        language: str = "en",
        output_dir: Optional[str] = None,
    ) -> Dict:
        """Generate both corrected and baseline audio for comparison.

        Paper Table 1 compares these to show zero forgetting on general speech.

        Args:
            text: Input text
            speaker_wav: Speaker reference
            language: Language code
            output_dir: Optional directory to save both versions

        Returns:
            Dict with both 'corrected' and 'baseline' results
        """
        corrected_path = None
        baseline_path = None

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            corrected_path = str(output_dir / "corrected.wav")
            baseline_path = str(output_dir / "baseline.wav")

        corrected = self.synthesize(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            output_path=corrected_path,
            return_gate_info=True,
        )

        baseline = self.synthesize_baseline(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            output_path=baseline_path,
        )

        return {
            "corrected": corrected,
            "baseline": baseline,
            "text": text,
        }

    def reload_memory(self, memory_path: str) -> None:
        """Hot-reload memory from disk without restarting.

        Useful for adding corrections while the inference server is running.

        Args:
            memory_path: Path to updated memory file
        """
        self._ensure_ready()
        self.memory.load(memory_path)
        logger.info(f"Memory reloaded: {self.memory.size} corrections")

    def _ensure_ready(self) -> None:
        """Ensure pipeline is loaded and ready."""
        if not self._is_ready:
            raise RuntimeError(
                "Inference pipeline not ready. Call inference.load() first."
            )
