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
            span_map_in_synth = []  # List of (span, new_c_start, new_c_end, p_text)
            if spans_with_phonetics:
                # Sort from end to beginning so character replacements don't shift earlier character offsets
                spans_sorted_desc = sorted(
                    retrieval.matched_spans,
                    key=lambda s: getattr(s, "char_start", 0) if getattr(s, "char_start", None) is not None else -1,
                    reverse=True
                )
                for span in spans_sorted_desc:
                    p_text = (getattr(span, "phonetic_text", None) or "").strip()
                    c_s = getattr(span, "char_start", None)
                    c_e = getattr(span, "char_end", None)
                    if p_text and p_text.lower() != span.word.lower() and c_s is not None and c_e is not None and 0 <= c_s < c_e <= len(synth_text):
                        synth_text = synth_text[:c_s] + p_text + synth_text[c_e:]
                        logger.info(f"[Hopfield Refiner] Applied span-specific pronunciation '{span.word}' → '{p_text}' at chars [{c_s}:{c_e}]")
                    elif p_text and p_text.lower() != span.word.lower():
                        pattern = r'\b' + re.escape(span.word) + r'\b'
                        synth_text = re.sub(pattern, p_text, synth_text, count=1, flags=re.IGNORECASE)
                        logger.info(f"[Hopfield Refiner] Applied pronunciation '{span.word}' → '{p_text}' in synthesis text")

            # Re-locate matched spans in synth_text for exact continuous delta alignment
            for span in retrieval.matched_spans:
                p_text = (getattr(span, "phonetic_text", None) or "").strip()
                search_word = p_text if (p_text and p_text.lower() in synth_text.lower()) else span.word
                pattern = re.escape(search_word)
                matches = [m.span() for m in re.finditer(pattern, synth_text, flags=re.IGNORECASE)]
                if matches:
                    # Pick match closest to original character start
                    orig_cs = getattr(span, "char_start", 0) or 0
                    best_m = min(matches, key=lambda m: abs(m[0] - orig_cs))
                    span_map_in_synth.append((span, best_m[0], best_m[1], search_word))
                else:
                    span_map_in_synth.append((span, getattr(span, "char_start", 0) or 0, getattr(span, "char_end", 0) or 0, span.word))

            scale = float(kwargs.get("correction_scale") or getattr(self.config, "correction_scale", 1.0))
            scaled_delta = retrieval.retrieved_delta.to(
                device=base_embeddings.device, dtype=base_embeddings.dtype
            ) * scale
            backbone_dim = getattr(backbone, "embedding_dim", 1024)

            # Construct continuous text_embedding_delta aligned with synth_text
            if backbone_dim == 512:
                # F5-TTS: character-level continuous DiT embedding space (d=512)
                delta_for_synth = torch.zeros(1, len(synth_text), backbone_dim, device=base_embeddings.device, dtype=base_embeddings.dtype)
                for span, s_start, s_end, word_text in span_map_in_synth:
                    target_len = max(1, s_end - s_start)
                    entry = self.memory.entries[span.entry_index] if span.entry_index < len(self.memory.entries) else None
                    if entry is not None:
                        w_delta = entry.word_delta if entry.word_delta is not None else entry.value.unsqueeze(0).unsqueeze(0)
                        w_delta = w_delta.to(device=base_embeddings.device, dtype=base_embeddings.dtype)
                        src_delta = w_delta[0] if w_delta.dim() == 3 else w_delta
                        # Adapt channel dimension to 512 if needed
                        if src_delta.shape[-1] != backbone_dim:
                            flat_src = src_delta.reshape(-1, 1, src_delta.shape[-1]).float()
                            res_src = F.interpolate(flat_src, size=backbone_dim, mode='linear', align_corners=True)
                            src_delta = res_src.reshape(list(src_delta.shape[:-1]) + [backbone_dim]).to(dtype=base_embeddings.dtype)
                        # Interpolate along sequence length to match target character span in synth_text
                        if src_delta.shape[0] != target_len:
                            if target_len == 1:
                                aligned_d = src_delta.mean(dim=0, keepdim=True)
                            elif src_delta.shape[0] == 1:
                                aligned_d = src_delta.repeat(target_len, 1)
                            else:
                                s_3d = src_delta.unsqueeze(0).transpose(1, 2)
                                aligned_d = F.interpolate(s_3d, size=target_len, mode='linear', align_corners=True).transpose(1, 2).squeeze(0)
                        else:
                            aligned_d = src_delta
                        actual_end = min(len(synth_text), s_start + target_len)
                        act_len = actual_end - s_start
                        if act_len > 0:
                            delta_for_synth[0, s_start:actual_end, :] = aligned_d[:act_len, :] * scale

                # Acoustic safety & clarity bounds for F5-TTS flow matching
                delta_for_synth = torch.clamp(delta_for_synth, -0.25, 0.25)
                active_mask = (delta_for_synth.abs().sum(dim=-1) > 1e-5)
                if active_mask.any():
                    d_norm = delta_for_synth[active_mask].norm().item()
                    max_f5_norm = 2.0
                    if d_norm > max_f5_norm:
                        delta_for_synth = delta_for_synth * (max_f5_norm / d_norm)
                        logger.info(f"[Hopfield Refiner] F5-TTS delta norm capped from {d_norm:.4f} to {max_f5_norm:.4f}")
            else:
                # XTTS / Other Backbones: token-level embedding space (d=1024)
                if synth_text == text:
                    delta_for_synth = scaled_delta
                else:
                    try:
                        synth_emb = backbone.encode_text(synth_text, language)
                        delta_for_synth = torch.zeros_like(synth_emb)
                        copy_len = min(scaled_delta.shape[1], synth_emb.shape[1])
                        delta_for_synth[:, :copy_len, :] = scaled_delta[:, :copy_len, :]
                    except Exception:
                        delta_for_synth = scaled_delta

                # Bounded relative delta norm for XTTS
                active_mask = (delta_for_synth.abs().sum(dim=-1) > 1e-5)
                if active_mask.any():
                    local_base_norm = base_embeddings[active_mask].norm().item() if active_mask.shape[1] == base_embeddings.shape[1] else 1.0
                    local_delta_norm = delta_for_synth[active_mask].norm().item()
                    alpha_cap = float(getattr(self.config, "max_relative_delta", 1.50))
                    max_allowed_norm = alpha_cap * local_base_norm
                    if local_delta_norm > max_allowed_norm and max_allowed_norm > 0:
                        delta_for_synth = delta_for_synth * (max_allowed_norm / local_delta_norm)
                        logger.info(f"[Hopfield Refiner] XTTS delta norm capped from {local_delta_norm:.4f} to {max_allowed_norm:.4f} (alpha={alpha_cap})")

            match_info = (
                [(s.word, s.start_pos, getattr(s, "phonetic_text", None)) for s in retrieval.matched_spans]
                if retrieval.matched_spans
                else [(retrieval.matched_word, 0, None)]
            )
            should_inject_delta = True
            logger.info(
                f"[Hopfield Refiner] INJECTION ACTIVE: {match_info} "
                f"(gate_max={retrieval.gate_values.max().item():.3f}, delta_norm={delta_for_synth.norm().item():.4f}, "
                f"inject_delta={should_inject_delta}, scale={scale}, synth_text='{synth_text}')."
            )

            waveform, sr = backbone.synthesize_direct(
                text=synth_text,
                speaker_conditioning=speaker_conditioning,
                language=language,
                user_ref_text=user_ref_text,
                text_embedding_delta=delta_for_synth,
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
