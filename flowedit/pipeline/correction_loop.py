"""
Correction Loop — Full FlowEdit Correction Pipeline.

Orchestrates the three stages of FlowEdit into a single workflow:
    Stage 1: Detection & Grounding (Whisper forced alignment)
    Stage 2: Latent Input Optimization (optimize δ*)
    Stage 3: Associative Memory Write (store in Hopfield memory)

Paper Figure 1 (Right): "Correction loop. User reference audio triggers
forced alignment, optimization of δ*, and a memory write to the Hopfield
memory."

Usage:
    loop = CorrectionLoop(config)
    loop.load_models()
    result = loop.correct(
        text="My friend Siobhan is visiting",
        target_word="Siobhan",
        ref_audio_path="./siobhan_correct.wav",
    )
"""

import torch
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path

from flowedit.config import FlowEditConfig
from flowedit.backbone import create_backbone
from flowedit.backbone.base import TTSBackbone
from flowedit.alignment.whisper_aligner import WhisperAligner, AlignmentResult
from flowedit.optimizer.latent_optimizer import LatentOptimizer, OptimizationResult
from flowedit.memory.hopfield_memory import HopfieldMemory

logger = logging.getLogger(__name__)


@dataclass
class CorrectionResult:
    """Result of a complete correction operation."""
    success: bool
    word: str
    alignment: Optional[AlignmentResult]
    optimization: Optional[OptimizationResult]
    memory_index: Optional[int]
    wall_clock_seconds: float
    memory_size: int
    error_message: Optional[str] = None


class CorrectionLoop:
    """Orchestrates the full FlowEdit correction pipeline."""

    def __init__(self, config: Optional[FlowEditConfig] = None):
        self.config = config or FlowEditConfig()

        # Component instances
        self.backbone: Optional[TTSBackbone] = None
        self.backbones: Dict[str, TTSBackbone] = {}
        self.aligner: Optional[WhisperAligner] = None
        self.optimizer: Optional[LatentOptimizer] = None
        self.memory: Optional[HopfieldMemory] = None

        self._models_loaded = False

    def get_backbone(self) -> TTSBackbone:
        """Get or lazily load the F5-TTS backbone instance."""
        key = "f5tts"
        if key in self.backbones:
            return self.backbones[key]

        logger.info(f"Loading requested backbone: {key.upper()}...")
        self.config.backbone.backbone_type = key
        bb = create_backbone(self.config.backbone)
        bb.load_model()
        self.backbones[key] = bb

        if self.backbone is None:
            self.backbone = bb

        return bb

    def load_models(self, memory_path: Optional[str] = None) -> None:
        """Load default models and initialize components."""
        logger.info("=" * 60)
        logger.info("Loading FlowEdit correction pipeline...")
        logger.info("=" * 60)

        start_time = time.time()

        # 1. Load primary backbone
        primary_type = "f5tts"
        logger.info(f"[1/4] Loading default backbone ({primary_type.upper()})...")
        self.backbone = self.get_backbone()

        # 2. Load Whisper aligner
        logger.info("[2/4] Loading Whisper aligner...")
        self.aligner = WhisperAligner(self.config.alignment)
        self.aligner.load_model()

        # 3. Initialize optimizer
        logger.info("[3/4] Initializing latent optimizer...")
        self.optimizer = LatentOptimizer(
            self.config.optimization,
            self.config.audio,
        )

        # 4. Initialize or load Hopfield memory
        logger.info("[4/4] Initializing Hopfield memory...")
        embed_dim = self.backbone.embedding_dim
        self.memory = HopfieldMemory(dim=embed_dim, config=self.config.memory)

        if memory_path and Path(memory_path).exists():
            try:
                self.memory.load(memory_path)
                logger.info(f"Loaded existing memory: {self.memory.size} corrections")
            except ValueError as e:
                logger.warning(f"Failed to load memory: {e}. Starting fresh.")

        elapsed = time.time() - start_time
        self._models_loaded = True

        logger.info("=" * 60)
        logger.info(f"FlowEdit pipeline ready in {elapsed:.1f}s")
        logger.info(f"  Primary Backbone: {primary_type.upper()} (dim={embed_dim})")
        logger.info(f"  Memory: {self.memory.size}/{self.config.memory.max_entries}")
        logger.info("=" * 60)

    def correct(
        self,
        text: str,
        target_word: str,
        ref_audio_path: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
    ) -> CorrectionResult:
        """Learn a pronunciation correction from reference audio.

        This runs the full FlowEdit correction loop:
        1. Whisper forced alignment to find target word boundaries
        2. Optimize perturbation δ* to match reference pronunciation
        3. Store correction in Hopfield memory

        Paper: "Corrections complete in approximately 15 seconds on a single GPU."

        Args:
            text: Text containing the target word (e.g., "My friend Siobhan")
            target_word: The word to correct (e.g., "Siobhan")
            ref_audio_path: Path to reference audio with correct pronunciation
            speaker_wav: Optional speaker reference for F5-TTS voice conditioning.
                         If None, uses the ref_audio as speaker reference too.
            language: Language code

        Returns:
            CorrectionResult with all pipeline outputs
        """
        self._ensure_loaded()
        bb = self.get_backbone()

        start_time = time.time()

        # Apply Indic phonetic normalizer (e.g. Mrunmayee -> Mroonmayee to prevent BPE 'Mr.' abbreviation distortion)
        from flowedit.utils.indic_phonetics import normalize_indic_phonetics
        text = normalize_indic_phonetics(text)
        
        # Isolate target_word if a multi-word string was passed (e.g. "Mrunmayee Sakharwade" -> "Mrunmayee")
        target_words_list = target_word.strip().split()
        primary_target_word = target_words_list[0] if target_words_list else target_word

        logger.info(f"\n{'='*60}")
        logger.info(f"CORRECTION: '{primary_target_word}' (full target: '{target_word}') in \"{text}\" (Backbone: F5TTS)")
        logger.info(f"Reference: {ref_audio_path}")
        logger.info(f"{'='*60}")

        try:
            # ════════════════════════════════════════════
            # Stage 1: Detection & Grounding
            # ════════════════════════════════════════════
            logger.info("\n▶ STAGE 1: Detection & Grounding (Whisper Alignment)")

            # Offload backbone to CPU while running Whisper aligner
            if bb is not None and getattr(bb, 'model', None) is not None and hasattr(bb.model, 'to'):
                bb.model.to("cpu")
                    
            whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.aligner is not None and getattr(self.aligner, '_model', None) is not None:
                if hasattr(self.aligner._model, 'to'):
                    self.aligner._model.to(whisper_device)
            torch.cuda.empty_cache()

            # Dynamically determine if the reference audio is just a single word
            import librosa
            audio_duration = librosa.get_duration(path=ref_audio_path)
            is_word_only = audio_duration < 2.0
            logger.info(f"Ref audio duration: {audio_duration:.2f}s, treating as word-only: {is_word_only}")

            alignment = self.aligner.align(
                audio_path=ref_audio_path,
                target_word=primary_target_word,
                full_text=None,
                language=language,
                ref_is_word_only=is_word_only,
            )

            # Map word boundaries to token indices using requested backbone tokenizer
            alignment = self.aligner.map_to_token_indices(
                alignment=alignment,
                full_text=text,
                target_word=primary_target_word,
                tokenizer=getattr(bb, "tokenizer", None) or getattr(bb.model, "tokenizer", None),
                language=language,
            )

            logger.info(
                f"  ✓ Aligned '{primary_target_word}' → tokens {alignment.token_indices} "
                f"({alignment.start_time:.2f}s - {alignment.end_time:.2f}s, "
                f"conf={alignment.confidence:.2f})"
            )

            if not alignment.token_indices:
                return CorrectionResult(
                    success=False,
                    word=target_word,
                    alignment=alignment,
                    optimization=None,
                    memory_index=None,
                    wall_clock_seconds=time.time() - start_time,
                    memory_size=self.memory.size,
                    error_message="No token indices found for target word",
                )

            # ════════════════════════════════════════════
            # Stage 2: Latent Input Optimization
            # ════════════════════════════════════════════
            logger.info(f"\n▶ STAGE 2: Latent Input Optimization via {bb.optimization_mode}")

            if self.aligner is not None and getattr(self.aligner, '_model', None) is not None:
                if hasattr(self.aligner._model, 'to'):
                    self.aligner._model.to("cpu")
                
            backbone_device = getattr(bb, 'device', "cuda" if torch.cuda.is_available() else "cpu")
            if bb is not None and getattr(bb, 'model', None) is not None and hasattr(bb.model, 'to'):
                bb.model.to(backbone_device)
            torch.cuda.empty_cache()

            if not speaker_wav:
                logger.info("speaker_wav not provided; defaulting to ref_audio_path for speaker conditioning.")
                speaker_path = ref_audio_path
            else:
                speaker_path = speaker_wav
            provided_ref_text = None
            
            speaker_conditioning = bb.get_speaker_embedding(
                speaker_path, 
                language,
                ref_text=provided_ref_text
            )

            # Phase 5a (Mandatory): Find fixed target sample indices from baseline synthesis
            import tempfile
            import os
            import soundfile as sf
            
            logger.info("\n▶ STAGE 1b: Target Audio Extraction (Baseline Alignment)")
            
            with torch.no_grad():
                baseline_wav, sr = bb.synthesize_direct(
                    text=text,
                    speaker_conditioning=speaker_conditioning,
                    language=language,
                    user_ref_text=provided_ref_text,
                    cfg_strength=2.0, # Must use a normal CFG so WhisperX can actually understand and align the audio
                )
                
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_base:
                tmp_base_path = tmp_base.name
            sf.write(tmp_base_path, baseline_wav.squeeze().cpu().numpy(), sr)
            
            # Align baseline synthesis
            whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.aligner is not None and getattr(self.aligner, '_model', None) is not None:
                if hasattr(self.aligner._model, 'to'):
                    self.aligner._model.to(whisper_device)
            torch.cuda.empty_cache()
            
            baseline_alignment = self.aligner.align(
                audio_path=tmp_base_path,
                target_word=primary_target_word,
                full_text=text,
                language=language,
                ref_is_word_only=False,
            )
            
            os.remove(tmp_base_path)
            
            # We only need start_time and end_time (sample boundaries) from baseline alignment, not token_indices.
                
            target_start_sample = int(baseline_alignment.start_time * sr)
            target_end_sample = int(baseline_alignment.end_time * sr)
            
            logger.info(f"  ✓ Target word in baseline synthesis: {baseline_alignment.start_time:.2f}s - {baseline_alignment.end_time:.2f}s "
                        f"(samples {target_start_sample}:{target_end_sample})")

            if self.aligner is not None and getattr(self.aligner, '_model', None) is not None:
                if hasattr(self.aligner._model, 'to'):
                    self.aligner._model.to("cpu")
            if bb is not None and getattr(bb, 'model', None) is not None and hasattr(bb.model, 'to'):
                bb.model.to(backbone_device)
            torch.cuda.empty_cache()

            optimization = self.optimizer.optimize(
                backbone=bb,
                text=text,
                target_word=primary_target_word,
                ref_audio_path=ref_audio_path,
                token_indices=alignment.token_indices,
                speaker_conditioning=speaker_conditioning,
                language=language,
                target_word_start_sample=target_start_sample,
                target_word_end_sample=target_end_sample,
            )

            logger.info(
                f"  ✓ Optimization complete: loss={optimization.final_loss:.4f}, "
                f"converged={optimization.converged}, "
                f"δ_target norm={torch.norm(optimization.delta_target).item():.4f}"
            )

            # ════════════════════════════════════════════
            # Stage 3: Memory Write
            # ════════════════════════════════════════════
            logger.info("\n▶ STAGE 3: Memory Write (Hopfield Network)")

            # Compute key: pool(c_I) — average text embedding of target tokens
            with torch.no_grad():
                base_embeddings = bb.encode_text(text, language)
                target_embeddings = base_embeddings[0, alignment.token_indices, :]
                key = target_embeddings.mean(dim=0)

                # Get context embeddings for homograph disambiguation
                context_start = max(0, min(alignment.token_indices) - self.config.memory.context_window)
                context_end = min(
                    base_embeddings.shape[1],
                    max(alignment.token_indices) + self.config.memory.context_window + 1
                )
                context_embeddings = base_embeddings[0, context_start:context_end, :]
                
                # The mean index of the target word within the context slice
                target_mean_idx = sum(alignment.token_indices) // len(alignment.token_indices)
                target_index_in_context = target_mean_idx - context_start

            # Value: pool(δ*_I) — average perturbation for target tokens
            value = optimization.delta_target.squeeze()

            memory_index = self.memory.write(
                key=key,
                value=value,
                word=target_word,
                context_embeddings=context_embeddings,
                target_index_in_context=target_index_in_context,
            )

            logger.info(
                f"  ✓ Stored at memory index {memory_index} "
                f"({self.memory.size}/{self.config.memory.max_entries})"
            )

            # ════════════════════════════════════════════
            # Summary
            # ════════════════════════════════════════════
            elapsed = time.time() - start_time

            logger.info(f"\n{'='*60}")
            logger.info(f"CORRECTION COMPLETE: '{target_word}'")
            logger.info(f"  Wall-clock time: {elapsed:.1f}s")
            logger.info(f"  Final loss: {optimization.final_loss:.4f}")
            logger.info(f"  Memory: {self.memory.size} corrections stored")
            logger.info(f"{'='*60}\n")

            return CorrectionResult(
                success=True,
                word=target_word,
                alignment=alignment,
                optimization=optimization,
                memory_index=memory_index,
                wall_clock_seconds=elapsed,
                memory_size=self.memory.size,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Correction failed for '{target_word}': {e}", exc_info=True)

            return CorrectionResult(
                success=False,
                word=target_word,
                alignment=None,
                optimization=None,
                memory_index=None,
                wall_clock_seconds=elapsed,
                memory_size=self.memory.size if self.memory else 0,
                error_message=str(e),
            )

    def save_memory(self, path: str) -> None:
        """Save current memory state to disk.

        Args:
            path: File path for memory persistence
        """
        self._ensure_loaded()
        self.memory.save(path)

    def list_corrections(self) -> List[Dict]:
        """List all stored corrections.

        Returns:
            List of correction metadata dicts
        """
        if self.memory is None:
            return []
        return self.memory.list_corrections()

    def _ensure_loaded(self) -> None:
        """Ensure all models are loaded."""
        if not self._models_loaded:
            raise RuntimeError(
                "Models not loaded. Call correction_loop.load_models() first."
            )
