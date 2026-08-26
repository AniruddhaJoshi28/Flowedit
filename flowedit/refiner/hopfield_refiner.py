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
            # If reference audio phonetic pronunciation is stored, substitute the target word
            for span in retrieval.matched_spans:
                p_text = getattr(span, "phonetic_text", None)
                if p_text and p_text.strip().lower() != span.word.strip().lower():
                    pattern = r'\b' + re.escape(span.word) + r'\b'
                    synth_text = re.sub(pattern, p_text.strip(), synth_text, flags=re.IGNORECASE)
                    logger.info(f"[Hopfield Refiner] Applied reference pronunciation conditioning: '{span.word}' → '{p_text.strip()}' in synthesis text")

            scale = float(kwargs.get("correction_scale") or getattr(self.config, "correction_scale", 1.0))
            scaled_delta = retrieval.retrieved_delta.to(
                device=base_embeddings.device, dtype=base_embeddings.dtype
            ) * scale

            # Safety constraint: allow full perturbation energy learned from reference audio
            active_mask = (scaled_delta.abs().sum(dim=-1) > 1e-5)
            if active_mask.any():
                local_base_norm = base_embeddings[active_mask].norm().item()
                local_delta_norm = scaled_delta[active_mask].norm().item()
                alpha_cap = float(getattr(self.config, "max_relative_delta", 1.50))
                max_allowed_norm = alpha_cap * local_base_norm
                if local_delta_norm > max_allowed_norm and max_allowed_norm > 0:
                    scaled_delta = scaled_delta * (max_allowed_norm / local_delta_norm)
                    logger.info(f"[Hopfield Refiner] Capped excessive delta norm from {local_delta_norm:.4f} to {max_allowed_norm:.4f} (alpha={alpha_cap})")

            match_info = (
                [(s.word, s.start_pos, getattr(s, "phonetic_text", None)) for s in retrieval.matched_spans]
                if retrieval.matched_spans
                else [(retrieval.matched_word, 0, None)]
            )
            logger.info(
                f"[Hopfield Refiner] INJECTION ACTIVE: {match_info} "
                f"(gate_max={retrieval.gate_values.max().item():.3f}, delta_norm={scaled_delta.norm().item():.4f}, scale={scale})."
            )

            waveform, sr = backbone.synthesize_direct(
                text=synth_text,
                speaker_conditioning=speaker_conditioning,
                language=language,
                user_ref_text=user_ref_text,
                text_embedding_delta=scaled_delta if synth_text == text else None,
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
