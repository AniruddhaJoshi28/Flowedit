"""
Hopfield Refiner — Inference-Time Conditioning Refinement for FlowEdit (arXiv:2606.20518).

Paper Section 3.2 & Equation 7:
    "During inference, text embeddings c are refined using the associative memory:
     c_hat = c + σ(max_j (β Q K_j^T) - τ) ⊙ Mem(Q)
     where Mem(Q) = softmax(β Q K^T) V, β = 1/√d, and τ is the learned similarity threshold."
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

        # Step 2: Query Hopfield Associative Memory (Eq. 6 & 7)
        retrieval = self.memory.retrieve(
            query_embeddings=base_embeddings,
            target_token_indices=target_token_indices,
        )

        # Step 3: Compute refined text conditioning c_hat = c + gate ⊙ Mem(Q) (Eq. 7)
        refined_embeddings = base_embeddings + retrieval.retrieved_delta

        # Step 4: Synthesize audio through Flow-Matching backbone
        if retrieval.is_active:
            logger.info(
                f"[Hopfield Refiner] Memory gate ACTIVE (max_gate={retrieval.gate_values.max().item():.3f}). "
                f"Applying retrieved pronunciation correction."
            )
            waveform, sr = backbone.synthesize_from_embeddings(
                text_embeddings=refined_embeddings,
                speaker_conditioning=speaker_conditioning,
                text=text,
                language=language,
                user_ref_text=user_ref_text,
                **kwargs,
            )
            is_modified = True
        else:
            logger.info("[Hopfield Refiner] Memory gate INACTIVE. Direct synthesis without modification.")
            waveform, sr = backbone.synthesize_direct(
                text=text,
                speaker_conditioning=speaker_conditioning,
                language=language,
                user_ref_text=user_ref_text,
                **kwargs,
            )
            is_modified = False

        return RefinementResult(
            waveform=waveform,
            sample_rate=sr,
            is_modified=is_modified,
            retrieval_result=retrieval,
            refined_embeddings=refined_embeddings,
            diagnostics={
                "num_memory_entries": self.memory.num_entries,
                "top_matches": retrieval.top_matches,
                "gate_max": retrieval.gate_values.max().item(),
            },
        )
