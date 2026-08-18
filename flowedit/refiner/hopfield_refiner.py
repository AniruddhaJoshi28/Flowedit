"""
Hopfield Refiner — Inference-Time Conditioning Refinement for FlowEdit (arXiv:2606.20518).

Paper Section 3.2 & Equation 7:
    "During inference, text embeddings c are refined using the associative memory:
     c_hat = c + σ(max_j (β Q K_j^T) - τ) ⊙ Mem(Q)
     where Mem(Q) = softmax(β Q K^T) V, β = 1/√d, and τ is the learned similarity threshold."

CRITICAL FIX (v3):
    The original implementation routed the delta through post-ConvNeXt embedding space
    (via encode_text → add → subtract → re-inject), which created a domain mismatch:
    δ* was optimized in pre-ConvNeXt space but injected after a lossy round-trip through
    non-linear ConvNeXt blocks. Additionally, mean-pooling destroyed per-character structure.

    The fix stores the FULL δ* tensor [1, S, d] from Stage 2 optimization and injects it
    directly at the inner nn.Embedding level during synthesis, preserving both the correct
    vector space and per-character spatial structure.
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

    def _find_word_occurrences(self, text: str, word: str) -> List[int]:
        """Find all character-level start indices of a word in text (case-insensitive)."""
        occurrences = []
        text_lower = text.lower()
        word_lower = word.lower()
        start = 0
        while True:
            idx = text_lower.find(word_lower, start)
            if idx == -1:
                break
            occurrences.append(idx)
            start = idx + 1
        return occurrences

    def _align_delta_to_text(
        self,
        full_delta: torch.Tensor,       # [1, S_orig, d]
        stored_word: str,
        stored_indices: List[int],
        stored_carrier: str,
        new_text: str,
        gate_values: Optional[torch.Tensor] = None,  # [1, S_new, 1] for homograph disambiguation
    ) -> Optional[torch.Tensor]:
        """Align stored per-character δ* to new synthesis text with contextual homograph disambiguation.

        Paper Section 3.2:
            "For homograph disambiguation (e.g., a 'bass' fish vs. 'bass' guitar),
             FlowEdit utilizes context-conditioned keys by taking a Gaussian-weighted
             average of surrounding text embeddings within a window of ±3 tokens."

        When the synthesis sentence contains homographs (e.g. "lead" as metal vs leader),
        the Hopfield gate scores σ(β Q K^T - τ) are high (>0.25) ONLY at the contextually
        matching occurrence and zero elsewhere. We apply the learned per-character δ*
        perturbation exclusively to the context-matching occurrence(s).
        """
        d = full_delta.shape[-1]
        device = full_delta.device
        dtype = full_delta.dtype
        new_len = len(new_text)

        # Fast path: if new_text is identical to stored_carrier, inject directly
        if new_text == stored_carrier and stored_indices:
            aligned = torch.zeros(1, new_len, d, device=device, dtype=dtype)
            idx_min = max(0, min(stored_indices))
            idx_max = min(full_delta.shape[1], max(stored_indices) + 1)
            end = min(new_len, idx_max)
            start = min(new_len, idx_min)
            copy_len = end - start
            if copy_len > 0:
                aligned[0, start:end, :] = full_delta[0, start:end, :].to(device=device, dtype=dtype)
            logger.info(
                f"[Delta Alignment] Exact match with carrier text. Direct δ* at [{start}:{end}] "
                f"(norm={aligned.norm().item():.4f})"
            )
            return aligned

        # Find all character-level occurrences of the word in new_text
        occurrences = self._find_word_occurrences(new_text, stored_word)
        if not occurrences:
            logger.warning(
                f"[Delta Alignment] Word '{stored_word}' not found in new text '{new_text}'."
            )
            return None

        word_len = len(stored_word)

        # Determine original word position in stored_carrier
        word_in_stored = stored_carrier.lower().find(stored_word.lower()) if stored_carrier else -1
        if word_in_stored >= 0 and stored_indices:
            w_start_orig = word_in_stored
            idx_min = min(stored_indices)
            idx_max = max(stored_indices) + 1
            pre_expansion = max(0, w_start_orig - idx_min)
            post_expansion = max(0, idx_max - (w_start_orig + word_len))
        elif stored_indices:
            w_start_orig = min(stored_indices)
            idx_min = min(stored_indices)
            idx_max = max(stored_indices) + 1
            pre_expansion = 0
            post_expansion = max(0, (idx_max - idx_min) - word_len)
        else:
            w_start_orig = 0
            idx_min = 0
            idx_max = full_delta.shape[1]
            pre_expansion = 0
            post_expansion = 0

        # --- Contextual Homograph Disambiguation (Paper Section 3.2) ---
        target_occurrences = []

        if len(occurrences) == 1:
            target_occurrences.append(occurrences[0])
            logger.info(
                f"[Delta Alignment] Single occurrence of '{stored_word}' at position {occurrences[0]}."
            )
        else:
            logger.info(
                f"[Delta Alignment] HOMOGRAPH DISAMBIGUATION: {len(occurrences)} occurrences of '{stored_word}' "
                f"at positions {occurrences}. Evaluating Hopfield gate scores per context..."
            )
            gate_flat = None
            if gate_values is not None:
                gate_flat = gate_values.squeeze()
                if gate_flat.dim() > 1:
                    gate_flat = gate_flat.squeeze(-1)

            best_occ = occurrences[0]
            best_score = -1.0

            for occ_start in occurrences:
                occ_end = min(occ_start + word_len, len(gate_flat) if gate_flat is not None else occ_start)
                if gate_flat is not None and occ_end > occ_start:
                    score = gate_flat[occ_start:occ_end].mean().item()
                else:
                    score = 0.0

                logger.info(f"  → Occurrence at [{occ_start}:{occ_start + word_len}] ('{new_text[occ_start:occ_start + word_len]}'): avg_gate={score:.4f}")
                if score > best_score:
                    best_score = score
                    best_occ = occ_start

                # If score exceeds active threshold (0.25), include this occurrence
                if score >= 0.25:
                    target_occurrences.append(occ_start)

            # Fallback: if none strictly exceeded 0.25, use the highest-scoring occurrence if best_score > 0.05
            if not target_occurrences:
                if best_score > 0.05:
                    target_occurrences.append(best_occ)
                    logger.info(f"  ✓ Disambiguated to best occurrence at position {best_occ} (gate_score={best_score:.4f})")
                else:
                    # Default to first occurrence if gate is uninformative
                    target_occurrences.append(occurrences[0])
                    logger.info(f"  ℹ Defaulting to first occurrence at position {occurrences[0]}")
            else:
                logger.info(f"  ✓ Disambiguated: Applying correction to context-matched occurrence(s) at {target_occurrences}")

        # Construct aligned delta tensor
        aligned = torch.zeros(1, new_len, d, device=device, dtype=dtype)

        for occ_start in target_occurrences:
            span_len = word_len + pre_expansion + post_expansion
            # Optional smooth Hann boundary taper to avoid edge discontinuities
            taper = torch.ones(span_len, device=device, dtype=dtype)
            if span_len > 4:
                taper[0] = 0.3
                taper[1] = 0.7
                taper[-2] = 0.7
                taper[-1] = 0.3

            # Map each character offset relative to word start
            for idx_p, p in enumerate(range(-pre_expansion, word_len + post_expansion)):
                orig_pos = w_start_orig + p
                new_pos = occ_start + p
                if 0 <= orig_pos < full_delta.shape[1] and 0 <= new_pos < new_len:
                    aligned[0, new_pos, :] = (full_delta[0, orig_pos, :].to(device=device, dtype=dtype)) * taper[idx_p]

        logger.info(
            f"[Delta Alignment] Successfully aligned δ* for '{stored_word}' across {len(target_occurrences)} occurrence(s). "
            f"Active span norm={aligned.norm().item():.4f}"
        )

        return aligned

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

        # Step 3: Decide synthesis path
        if retrieval.is_active and retrieval.matched_full_delta is not None:
            # ─── DIRECT INJECTION PATH (correct: pre-ConvNeXt space) ───
            # Align stored full δ* to the current synthesis text
            aligned_delta = self._align_delta_to_text(
                full_delta=retrieval.matched_full_delta,
                stored_word=retrieval.matched_word,
                stored_indices=retrieval.matched_token_indices or [],
                stored_carrier=retrieval.matched_carrier_text or "",
                new_text=text,
                gate_values=retrieval.gate_values,
            )

            if aligned_delta is not None:
                scale = getattr(self.config, "correction_scale", 1.0)
                scaled_delta = aligned_delta * scale

                logger.info(
                    f"[Hopfield Refiner] DIRECT INJECTION: '{retrieval.matched_word}' "
                    f"(gate_max={retrieval.gate_values.max().item():.3f}, "
                    f"delta_norm={scaled_delta.norm().item():.4f}, scale={scale}). "
                    f"Bypassing encode→add→subtract chain."
                )

                # Synthesize with δ* injected directly at inner nn.Embedding level
                waveform, sr = backbone.synthesize_direct(
                    text=text,
                    speaker_conditioning=speaker_conditioning,
                    language=language,
                    user_ref_text=user_ref_text,
                    text_embedding_delta=scaled_delta,
                    **kwargs,
                )

                # For diagnostics, compute what refined_embeddings would be
                # (not used for synthesis, just for the return value)
                refined_embeddings = base_embeddings  # placeholder

                return RefinementResult(
                    waveform=waveform,
                    sample_rate=sr,
                    is_modified=True,
                    retrieval_result=retrieval,
                    refined_embeddings=refined_embeddings,
                    diagnostics={
                        "num_memory_entries": self.memory.num_entries,
                        "top_matches": retrieval.top_matches,
                        "gate_max": retrieval.gate_values.max().item(),
                        "injection_mode": "direct_full_delta",
                        "delta_norm": scaled_delta.norm().item(),
                    },
                )

            # Fall through to legacy path if alignment failed
            logger.warning("[Hopfield Refiner] Delta alignment failed. Falling back to legacy path.")

        if retrieval.is_active:
            # ─── LEGACY FALLBACK PATH (pooled delta via synthesize_from_embeddings) ───
            logger.info(
                f"[Hopfield Refiner] Memory gate ACTIVE but no full_delta. "
                f"Using legacy pooled delta path (max_gate={retrieval.gate_values.max().item():.3f})."
            )
            scale = getattr(self.config, "correction_scale", 1.5)
            raw_delta = retrieval.retrieved_delta.to(device=base_embeddings.device, dtype=base_embeddings.dtype)
            retrieved_delta = raw_delta * scale
            refined_embeddings = base_embeddings + retrieved_delta

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
            # ─── NO CORRECTION PATH ───
            logger.info("[Hopfield Refiner] Memory gate INACTIVE. Direct synthesis without modification.")
            refined_embeddings = base_embeddings
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
                "injection_mode": "legacy_pooled" if is_modified else "none",
            },
        )
