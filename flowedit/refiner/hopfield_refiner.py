"""
Hopfield Refiner — Inference-Time Conditioning Refinement for FlowEdit (arXiv:2606.20518).

Paper Section 3.2 & Equation 7:
    "During inference, text embeddings c are refined using the associative memory:
     c_hat = c + σ(max_j (β Q K_j^T) - τ) ⊙ Mem(Q)
     where Mem(Q) = softmax(β Q K^T) V, β = 1/√d, and τ is the learned similarity threshold."

Enhanced for:
    1. Multi-word simultaneous correction injection across complete sentences.
    2. Contextual homograph disambiguation (accurate per-span injection).
    3. Speaker-invariant direct synthesis without audio degradation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn

from flowedit.config import MemoryConfig
from flowedit.memory.hopfield_memory import HopfieldMemory, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class RefinementResult:
    """Structured result of inference refinement."""
    waveform: torch.Tensor          # Output synthesized waveform [1, T]
    sample_rate: int                # Audio sample rate (24000)
    is_modified: bool               # True if memory gate was active
    retrieval_result: RetrievalResult
    refined_embeddings: torch.Tensor
    diagnostics: Dict[str, Any]


class HopfieldRefiner(nn.Module):
    """Refines text embeddings c at inference time using Hopfield Associative Memory."""

    def __init__(self, memory: HopfieldMemory, config: Optional[MemoryConfig] = None):
        super().__init__()
        self.memory = memory
        self.config = config or MemoryConfig()

    def forward(
        self,
        backbone,
        text: str,
        speaker_conditioning: Dict[str, Any],
        language: str = "en",
        target_token_indices: Optional[List[int]] = None,
        user_ref_text: Optional[str] = None,
        **kwargs,
    ) -> RefinementResult:
        """Refine text embeddings and synthesize audio (Paper Section 3.2 & Eq. 7).

        Args:
            backbone: TTSBackbone instance
            text: Input text string to synthesize
            speaker_conditioning: Speaker conditioning dictionary
            language: Language code
            target_token_indices: Optional target token span
            user_ref_text: Optional reference text override

        Returns:
            RefinementResult containing synthesized waveform and diagnostics
        """
        # Step 1: Base text embedding c = E(x) ∈ R^[1, S, d]
        base_embeddings = backbone.encode_text(text, language)

        # Step 2: Query Hopfield Associative Memory with text context (Eq. 6 & 7)
        retrieval = self.memory.retrieve(
            query_embeddings=base_embeddings,
            target_token_indices=target_token_indices,
            text=text,
            tokenizer=getattr(backbone, "tokenizer_instance", None),
            language=language,
        )

        # Step 3: Decide synthesis path
        has_active_delta = retrieval.is_active and (
            retrieval.retrieved_delta.abs().sum() > 1e-5 or any(getattr(s, "phonetic_text", None) for s in retrieval.matched_spans)
        )

        if has_active_delta:
            import re
            synth_text = text

            # Resolve phonetic pronunciation for matched spans if not explicitly stored
            from flowedit.alignment.homograph_resolver import HomographContextResolver
            resolver = HomographContextResolver()

            for span in retrieval.matched_spans:
                p_text = getattr(span, "phonetic_text", None)
                if not p_text or p_text.strip().lower() == span.word.strip().lower():
                    # 1. Check entry stored phonetic_text or metadata
                    if span.entry_index < len(self.memory.entries):
                        entry = self.memory.entries[span.entry_index]
                        c_p = getattr(entry, "phonetic_text", None) or entry.metadata.get("phonetic_text") or entry.metadata.get("phonetic_hint")
                        if c_p and c_p.strip().lower() != span.word.strip().lower():
                            span.phonetic_text = c_p.strip()
                            continue
                        # 2. Check contextual homograph resolver against entry carrier_text
                        h_p = resolver.resolve_homograph_phonetic(span.word, entry.carrier_text or text)
                        if h_p and h_p.strip().lower() != span.word.strip().lower():
                            span.phonetic_text = h_p.strip()
                            continue
                    # 3. Check contextual homograph resolver against full input text
                    h_p = resolver.resolve_homograph_phonetic(span.word, text)
                    if h_p and h_p.strip().lower() != span.word.strip().lower():
                        span.phonetic_text = h_p.strip()

            # If reference audio phonetic pronunciation is stored, substitute only the matched spans
            spans_with_phonetics = [
                s for s in retrieval.matched_spans
                if getattr(s, "phonetic_text", None) and s.phonetic_text.strip().lower() != s.word.strip().lower()
            ]
            if spans_with_phonetics:
                # Sort from end to beginning so character replacements don't shift earlier character offsets
                spans_with_phonetics = sorted(
                    spans_with_phonetics,
                    key=lambda s: getattr(s, "char_start", 0) if getattr(s, "char_start", None) is not None else -1,
                    reverse=True
                )
                for span in spans_with_phonetics:
                    p_text = span.phonetic_text.strip()
                    c_s = getattr(span, "char_start", None)
                    c_e = getattr(span, "char_end", None)
                    if c_s is not None and c_e is not None and 0 <= c_s < c_e <= len(synth_text):
                        synth_text = synth_text[:c_s] + p_text + synth_text[c_e:]
                        logger.info(f"[Hopfield Refiner] Applied span-specific pronunciation '{span.word}' → '{p_text}' at chars [{c_s}:{c_e}]")
                    else:
                        pattern = r'\b' + re.escape(span.word) + r'\b'
                        synth_text = re.sub(pattern, p_text, synth_text, count=1, flags=re.IGNORECASE)
                        logger.info(f"[Hopfield Refiner] Applied pronunciation '{span.word}' → '{p_text}' in synthesis text")

            scale = float(kwargs.get("correction_scale") or getattr(self.config, "correction_scale", 1.0))
            scaled_delta = retrieval.retrieved_delta.to(
                device=base_embeddings.device, dtype=base_embeddings.dtype
            ) * scale

            # Check backbone space compatibility:
            # XTTS continuous delta (learned in d=1024 discrete space) must not be injected into F5-TTS DiT (d=512)
            # because coordinates across different architectures destroy flow-matching trajectories into noise.
            backbone_dim = getattr(backbone, "embedding_dim", 1024)
            is_backbone_compatible = True
            for span in retrieval.matched_spans:
                if span.entry_index < len(self.memory.entries):
                    entry = self.memory.entries[span.entry_index]
                    entry_dim = entry.key.shape[-1] if hasattr(entry, "key") else 1024
                    if entry_dim != backbone_dim and not getattr(span, "phonetic_text", None):
                        is_backbone_compatible = False
                        logger.info(
                            f"[Hopfield Refiner] Entry '{span.word}' was trained in {entry_dim}d space. "
                            f"Current backbone uses {backbone_dim}d space. Bypassing continuous delta to preserve audio clarity."
                        )
                        break

            # Safety constraint: bound perturbation norm
            active_mask = (scaled_delta.abs().sum(dim=-1) > 1e-5)
            if active_mask.any():
                local_base_norm = base_embeddings[active_mask].norm().item()
                local_delta_norm = scaled_delta[active_mask].norm().item()
                alpha_cap = 0.30 if backbone_dim == 512 else float(getattr(self.config, "max_relative_delta", 1.50))
                max_allowed_norm = alpha_cap * local_base_norm
                if local_delta_norm > max_allowed_norm and max_allowed_norm > 0:
                    scaled_delta = scaled_delta * (max_allowed_norm / local_delta_norm)
                    logger.info(f"[Hopfield Refiner] Capped delta norm from {local_delta_norm:.4f} to {max_allowed_norm:.4f} (alpha={alpha_cap})")

            match_info = (
                [(s.word, s.start_pos, getattr(s, "phonetic_text", None)) for s in retrieval.matched_spans]
                if retrieval.matched_spans
                else [(retrieval.matched_word, 0, None)]
            )
            should_inject_delta = (synth_text == text) and is_backbone_compatible
            logger.info(
                f"[Hopfield Refiner] INJECTION ACTIVE: {match_info} "
                f"(gate_max={retrieval.gate_values.max().item():.3f}, delta_norm={scaled_delta.norm().item():.4f}, "
                f"inject_delta={should_inject_delta}, scale={scale})."
            )

            waveform, sr = backbone.synthesize_direct(
                text=synth_text,
                speaker_conditioning=speaker_conditioning,
                language=language,
                user_ref_text=user_ref_text,
                text_embedding_delta=scaled_delta if should_inject_delta else None,
                **kwargs,
            )

            return RefinementResult(
                waveform=waveform,
                sample_rate=sr,
                is_modified=True,
                retrieval_result=retrieval,
                refined_embeddings=base_embeddings + scaled_delta,
                diagnostics={
                    "num_memory_entries": self.memory.num_entries,
                    "top_matches": retrieval.top_matches,
                    "matched_spans": [
                        {
                            "word": s.word,
                            "start": s.start_pos,
                            "end": s.end_pos,
                            "score": s.score,
                            "gate": s.gate_value,
                            "phonetic_text": getattr(s, "phonetic_text", None),
                        }
                        for s in retrieval.matched_spans
                    ],
                    "gate_max": retrieval.gate_values.max().item(),
                    "injection_mode": "multi_span_word_delta",
                    "delta_norm": scaled_delta.norm().item(),
                    "synth_text": synth_text,
                },
            )
        else:
            logger.info("[Hopfield Refiner] Memory gate INACTIVE. Direct baseline synthesis.")
            waveform, sr = backbone.synthesize_direct(
                text=text,
                speaker_conditioning=speaker_conditioning,
                language=language,
                user_ref_text=user_ref_text,
                text_embedding_delta=None,
                **kwargs,
            )

            return RefinementResult(
                waveform=waveform,
                sample_rate=sr,
                is_modified=False,
                retrieval_result=retrieval,
                refined_embeddings=base_embeddings,
                diagnostics={
                    "num_memory_entries": self.memory.num_entries,
                    "top_matches": retrieval.top_matches,
                    "gate_max": retrieval.gate_values.max().item() if retrieval.gate_values.numel() > 0 else 0.0,
                    "injection_mode": "none",
                },
            )
