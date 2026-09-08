"""
Correction Loop — Full FlowEdit Correction Pipeline (arXiv:2606.20518).

Orchestrates the three stages of FlowEdit:
    Stage 1: Detection & Grounding (Whisper forced alignment)
    Stage 2: Latent Input Optimization (50-step Adam optimization of perturbation δ*)
    Stage 3: Associative Memory Storage (Continuous Modern Hopfield Network with cluster averaging)
"""

import time
import os
import tempfile
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Union
from pathlib import Path

import torch

try:
    import soundfile as sf
except ImportError:
    sf = None
try:
    import librosa
except ImportError:
    librosa = None

from flowedit.config import FlowEditConfig
from flowedit.backbone import create_backbone
from flowedit.backbone.base import TTSBackbone
from flowedit.alignment.whisper_aligner import WhisperAligner, AlignmentResult
from flowedit.optimizer.latent_optimizer import LatentOptimizer, OptimizationResult
from flowedit.memory.hopfield_memory import HopfieldMemory
from flowedit.memory.candidate_validator import CandidateValidator
from flowedit.utils.indic_phonetics import normalize_indic_phonetics

logger = logging.getLogger(__name__)


@dataclass
class CorrectionResult:
    """Result of a complete FlowEdit correction operation."""
    success: bool
    word: str
    alignment: Optional[AlignmentResult]
    optimization: Optional[OptimizationResult]
    memory_index: Optional[int]
    wall_clock_seconds: float
    memory_size: int
    error_message: Optional[str] = None


class CorrectionLoop:
    """Orchestrates the end-to-end FlowEdit correction pipeline."""

    def __init__(self, config: Optional[FlowEditConfig] = None):
        self.config = config or FlowEditConfig()
        self.backbone: Optional[TTSBackbone] = None
        self.aligner: Optional[WhisperAligner] = None
        self.optimizer: Optional[LatentOptimizer] = None
        self.memory: Optional[HopfieldMemory] = None
        self._models_loaded = False

    def load_models(self) -> None:
        """Initialize and load all FlowEdit models."""
        logger.info("=" * 60)
        logger.info("Initializing FlowEdit Pipeline (arXiv:2606.20518)...")
        logger.info("=" * 60)
        t0 = time.time()

        # 1. Flow-Matching DiT Backbone
        self.backbone = create_backbone(self.config.backbone)
        self.backbone.load_model()

        # 2. Whisper Aligner
        self.aligner = WhisperAligner(self.config.alignment)
        self.aligner.load_model()

        # 3. Latent Optimizer
        self.optimizer = LatentOptimizer(self.config.optimization, self.config.audio)

        # 4. Modern Hopfield Memory
        embed_dim = self.backbone.embedding_dim
        self.memory = HopfieldMemory(self.config.memory, embedding_dim=embed_dim)

        self._models_loaded = True
        logger.info(f"✓ FlowEdit Pipeline ready in {time.time() - t0:.2f}s (Embedding dim d={embed_dim})")

    def _ensure_loaded(self) -> None:
        if not self._models_loaded:
            self.load_models()

    def correct(
        self,
        text: str,
        target_word: str,
        ref_audio_path: str,
        speaker_wav: Union[str, List[str]],
        language: str = "en",
        user_ref_text: Optional[str] = None,
        occurrence_index: int = 0,
        extra_speaker_wavs: Optional[List[str]] = None,
        phonetic_hint: Optional[str] = None,
    ) -> CorrectionResult:
        """Learn and store a pronunciation correction from reference audio.

        Args:
            text: Full carrier sentence containing target word
            target_word: Target word to correct
            ref_audio_path: Path to audio with target pronunciation
            speaker_wav: Speaker voice audio for conditioning (single path or list of paths)
            language: Language code
            user_ref_text: Optional reference text
            occurrence_index: Index if target word occurs multiple times
            extra_speaker_wavs: Optional additional speaker audio paths for multi-speaker joint optimization

        Returns:
            CorrectionResult
        """
        self._ensure_loaded()
        t_start = time.time()

        # Normalize text and target word
        text = normalize_indic_phonetics(text)
        clean_target = target_word.strip()
        if not clean_target:
            raise ValueError("target_word cannot be empty.")

        primary_target_word = clean_target.split()[0]
        if primary_target_word.lower() not in text.lower():
            raise ValueError(f"Target word '{primary_target_word}' not found in sentence: '{text}'")

        logger.info(f"\n{'='*60}")
        logger.info(f"FlowEdit Correction: '{primary_target_word}' in \"{text}\"")
        logger.info(f"Ref Audio: {ref_audio_path} | Speaker: {speaker_wav}")
        logger.info(f"{'='*60}")

        try:
            # ─────────────────────────────────────────────────────────────
            # Stage 1: Detection & Grounding (Whisper Alignment)
            # ─────────────────────────────────────────────────────────────
            logger.info("▶ Stage 1: Whisper Forced Alignment")
            # Determine if reference audio is word-level (≤ 6.0s or short ref text)
            ref_dur = librosa.get_duration(path=ref_audio_path)
            is_word_only = (user_ref_text is None or len(user_ref_text.split()) <= 2) and (ref_dur < 6.0)

            alignment = self.aligner.align(
                audio_path=ref_audio_path,
                target_word=primary_target_word,
                full_text=user_ref_text if not is_word_only else None,
                language=language,
                ref_is_word_only=is_word_only,
            )

            alignment = self.aligner.map_to_token_indices(
                alignment=alignment,
                full_text=text,
                target_word=primary_target_word,
                tokenizer=self.backbone.tokenizer,
                language=language,
                occurrence_index=occurrence_index,
                target_phrase=clean_target if clean_target != primary_target_word else None,
            )


            logger.info(
                f"  ✓ Aligned '{primary_target_word}' → Tokens I: {alignment.token_indices} "
                f"({alignment.start_time:.2f}s - {alignment.end_time:.2f}s, Conf={alignment.confidence:.2f})"
            )

            if not alignment.token_indices:
                return CorrectionResult(
                    success=False,
                    word=primary_target_word,
                    alignment=alignment,
                    optimization=None,
                    memory_index=None,
                    wall_clock_seconds=time.time() - t_start,
                    memory_size=self.memory.num_entries,
                    error_message="Could not resolve token indices for target word",
                )

            # Explicit user phonetic hint (Whisper auto-hallucinations disabled)
            if phonetic_hint:
                logger.info(f"  o- Using explicit user phonetic hint: '{phonetic_hint}'")

            # Collect speaker conditionings for multi-speaker joint optimization
            speaker_paths: List[str] = []
            if isinstance(speaker_wav, list):
                speaker_paths.extend(speaker_wav)
            elif speaker_wav:
                speaker_paths.append(speaker_wav)

            if extra_speaker_wavs:
                for p in extra_speaker_wavs:
                    if p and p not in speaker_paths:
                        speaker_paths.append(p)

            speaker_cond_list = [
                self.backbone.get_speaker_embedding(p, language=language, ref_text=user_ref_text)
                for p in speaker_paths
            ]
            speaker_conditioning = speaker_cond_list if len(speaker_cond_list) > 1 else speaker_cond_list[0]

            # ─────────────────────────────────────────────────────────────
            # Stage 2: Latent Input Optimization
            # ─────────────────────────────────────────────────────────────
            logger.info("▶ Stage 2: Latent Input Optimization (50 Adam steps)")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            optimization = self.optimizer.optimize(
                backbone=self.backbone,
                text=text,
                target_word=primary_target_word,
                ref_audio_path=ref_audio_path,
                token_indices=alignment.token_indices,
                speaker_conditioning=speaker_conditioning,
                language=language,
                ref_start_time=getattr(alignment, "start_time", None),
                phonetic_hint=phonetic_hint,
                ref_end_time=getattr(alignment, "end_time", None),
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Candidate Validation Gate
            validator = CandidateValidator(max_relative_delta=self.config.optimization.max_relative_delta)
            val_res = validator.validate(
                optimization_result=optimization,
                word=primary_target_word,
                initial_loss=optimization.initial_loss,
                final_loss=optimization.final_loss,
            )

            if not val_res.accepted:
                logger.warning(f"  ✗ Correction REJECTED: {val_res.reason}")
                return CorrectionResult(
                    success=False,
                    word=primary_target_word,
                    alignment=alignment,
                    optimization=optimization,
                    memory_index=None,
                    wall_clock_seconds=time.time() - t_start,
                    memory_size=self.memory.num_entries,
                    error_message=val_res.reason,
                )

            # ─────────────────────────────────────────────────────────────
            # Stage 3: Associative Memory Storage & Cluster Averaging
            # ─────────────────────────────────────────────────────────────
            logger.info("▶ Stage 3: Modern Hopfield Associative Memory Write / Cluster Average")
            effective_phonetic = phonetic_hint or getattr(alignment, "auto_phonetic_hint", None) or clean_target
            if effective_phonetic and effective_phonetic.strip().lower() != primary_target_word.lower():
                logger.info(f"  ✓ Reference audio pronunciation representation: '{effective_phonetic}'")

            with torch.no_grad():
                base_embeddings = (
                    optimization.base_embeddings
                    if getattr(optimization, "base_embeddings", None) is not None
                    else self.backbone.encode_text(text, language)
                )
                key = self.memory.compute_context_key(
                    base_embeddings,
                    alignment.token_indices,
                    carrier_text=text,
                )

            value = optimization.delta_pooled

            mem_idx, action = self.memory.write(
                key=key,
                value=value,
                word=primary_target_word,
                carrier_text=text,
                token_indices=alignment.token_indices,
                language=language,
                full_delta=optimization.delta,
                word_delta=getattr(optimization, "word_delta", None),
                phonetic_text=effective_phonetic,
            )

            elapsed = time.time() - t_start
            logger.info(f"\n{'='*60}")
            logger.info(f"✓ FLOWEDIT CORRECTION SUCCESS: '{primary_target_word}' in {elapsed:.2f}s")
            logger.info(f"  Memory Action: {action.upper()} at index {mem_idx} | Total Entries: {self.memory.num_entries}")
            logger.info(f"{'='*60}\n")

            return CorrectionResult(
                success=True,
                word=primary_target_word,
                alignment=alignment,
                optimization=optimization,
                memory_index=mem_idx,
                wall_clock_seconds=elapsed,
                memory_size=self.memory.num_entries,
            )

        except Exception as e:
            elapsed = time.time() - t_start
            logger.error(f"Correction failed for '{primary_target_word}': {e}", exc_info=True)
            return CorrectionResult(
                success=False,
                word=primary_target_word,
                alignment=None,
                optimization=None,
                memory_index=None,
                wall_clock_seconds=elapsed,
                memory_size=self.memory.num_entries if self.memory else 0,
                error_message=str(e),
            )
