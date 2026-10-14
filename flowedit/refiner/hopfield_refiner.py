"""
Hopfield Refiner — Inference-Time Conditioning Refinement for FlowEdit (arXiv:2606.20518).

Paper Section 3.2 & Equation 7:
    "During inference, text embeddings c are refined using the associative memory:
     c_hat = c + σ(max_j (β Q K_j^T) - τ) ⊙ Mem(Q)
     where Mem(Q) = softmax(β Q K^T) V, β = 1/√d, and τ is the learned similarity threshold."

Enhanced for:
    1. Multi-word simultaneous correction injection across complete sentences.
    2. Per-occurrence homograph pronunciation: every matched span is respelled with the
       pronunciation of the sense cluster the memory assigned to that occurrence.
    3. Backbone-declared alignment (character vs token level) and acoustic safety bounds.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowedit.config import MemoryConfig
from flowedit.memory.hopfield_memory import HopfieldMemory, MatchedWordSpan, RetrievalResult

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

    @staticmethod
    def _apply_pronunciations(
        text: str, spans: List[MatchedWordSpan]
    ) -> Tuple[str, List[Tuple[MatchedWordSpan, int, int]]]:
        """Respell each matched occurrence with its own sense's pronunciation.

        Returns the synthesis text and, for every matched span, its [start, end) in that text.
        Offsets follow exactly from the substitutions (no re-searching), so repeated words keep
        their own positions.
        """
        def match_case(original: str, spelling: str) -> str:
            if original.isupper() and len(original) > 1:
                return spelling.upper()
            if original[:1].isupper():
                return spelling[:1].upper() + spelling[1:]
            return spelling

        edits = []
        for sp in spans:
            if sp.char_start is None or sp.char_end is None:
                continue
            p = (sp.phonetic_text or "").strip()
            original = text[sp.char_start:sp.char_end]
            new = match_case(original, p) if p and p.lower() != sp.word.strip().lower() else original
            edits.append((sp.char_start, sp.char_end, new, sp))
        edits.sort(key=lambda e: e[0])

        out, located, cursor, shift = [], [], 0, 0
        for c_s, c_e, new, sp in edits:
            if c_s < cursor:
                continue  # overlapping spans: keep the first
            out.append(text[cursor:c_s])
            out.append(new)
            located.append((sp, c_s + shift, c_s + shift + len(new)))
            if new != text[c_s:c_e]:
                logger.info(f"[Hopfield Refiner] '{text[c_s:c_e]}'@{c_s} -> '{new}'")
            shift += len(new) - (c_e - c_s)
            cursor = c_e
        out.append(text[cursor:])
        return "".join(out), located

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

        # Step 3: Each matched span already carries its own sense (chosen per occurrence by the memory)
        spans_with_phonetics = [
            sp for sp in retrieval.matched_spans
            if sp.phonetic_text and sp.phonetic_text.strip().lower() != sp.word.strip().lower()
            and sp.char_start is not None and sp.char_end is not None
        ]
        has_active_delta = retrieval.is_active and (
            retrieval.retrieved_delta.abs().sum() > 1e-5 or bool(spans_with_phonetics)
        )

        if has_active_delta:
            # Substitute learned pronunciations span-by-span and track where every matched span lands
            synth_text, synth_spans = self._apply_pronunciations(text, retrieval.matched_spans)

            scale = float(kwargs.get("correction_scale") or self.config.correction_scale)
            scaled_delta = retrieval.retrieved_delta.to(
                device=base_embeddings.device, dtype=base_embeddings.dtype
            ) * scale
            backbone_dim = backbone.embedding_dim

            if getattr(backbone, "is_character_level", False):
                # Character-level backbone (F5-TTS): one perturbation row per character of synth_text
                delta_for_synth = torch.zeros(1, len(synth_text), backbone_dim, device=base_embeddings.device, dtype=base_embeddings.dtype)
                for span, s_start, s_end in synth_spans:
                    entry = self.memory.entries[span.entry_index] if 0 <= span.entry_index < len(self.memory.entries) else None
                    if entry is None or not span.has_delta or entry.word_delta is None:
                        continue
                    w_delta = entry.word_delta.to(device=base_embeddings.device, dtype=base_embeddings.dtype)
                    src_delta = w_delta[0] if w_delta.dim() == 3 else w_delta
                    target_len = max(1, s_end - s_start)
                    aligned_d = self.memory._align_delta(src_delta, target_len)
                    actual_end = min(len(synth_text), s_start + target_len)
                    if actual_end > s_start:
                        delta_for_synth[0, s_start:actual_end, :] = aligned_d[:actual_end - s_start, :] * scale
            elif synth_text == text:
                delta_for_synth = scaled_delta
            else:
                # Token-level backbone (XTTS) with respelled words: re-align δ* to synth_text tokens
                synth_emb = backbone.encode_text(synth_text, language)
                delta_for_synth = torch.zeros_like(synth_emb)
                S_synth = synth_emb.shape[1]
                for span, s_start, s_end in synth_spans:
                    if not span.has_delta:
                        continue
                    entry = self.memory.entries[span.entry_index] if 0 <= span.entry_index < len(self.memory.entries) else None
                    w_delta = entry.word_delta if (entry is not None and entry.word_delta is not None) else scaled_delta[:, span.start_pos:span.end_pos, :]
                    if w_delta.numel() == 0:
                        continue
                    w_delta = w_delta.to(device=base_embeddings.device, dtype=base_embeddings.dtype)
                    src_delta = w_delta[0] if w_delta.dim() == 3 else w_delta
                    tok_s, tok_e = self.memory._char_span_to_token_span(s_start, s_end, len(synth_text), S_synth)
                    target_tok_len = max(1, tok_e - tok_s)
                    aligned_d = self.memory._align_delta(src_delta, target_tok_len)
                    act_end = min(S_synth, tok_s + target_tok_len)
                    if act_end > tok_s:
                        delta_for_synth[0, tok_s:act_end, :] = aligned_d[:act_end - tok_s, :] * scale

            # Acoustic safety bounds declared by the backbone (element clamp + L2 cap)
            clamp_val, norm_cap = getattr(backbone, "delta_bounds", (float("inf"), float("inf")))
            if clamp_val != float("inf"):
                delta_for_synth = torch.clamp(delta_for_synth, -clamp_val, clamp_val)
            active_mask = (delta_for_synth.abs().sum(dim=-1) > 1e-5)
            if active_mask.any():
                d_norm = delta_for_synth[active_mask].norm().item()
                if d_norm > norm_cap:
                    delta_for_synth = delta_for_synth * (norm_cap / d_norm)
                    logger.info(f"[Hopfield Refiner] delta norm capped from {d_norm:.4f} to {norm_cap:.4f}")
            else:
                delta_for_synth = None

            match_info = (
                [(s.word, s.start_pos, getattr(s, "phonetic_text", None)) for s in retrieval.matched_spans]
                if retrieval.matched_spans
                else [(retrieval.matched_word, 0, None)]
            )
            logger.info(
                f"[Hopfield Refiner] INJECTION ACTIVE: {match_info} "
                f"(gate_max={retrieval.gate_values.max().item():.3f}, "
                f"delta_norm={(delta_for_synth.norm().item() if delta_for_synth is not None else 0.0):.4f}, "
                f"scale={scale}, synth_text='{synth_text}')."
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
                            "char_start": s.char_start,
                            "sense_similarity": s.sense_similarity,
                            "has_delta": s.has_delta,
                        }
                        for s in retrieval.matched_spans
                    ],
                    "gate_max": retrieval.gate_values.max().item(),
                    "injection_mode": "multi_span_word_delta" if delta_for_synth is not None else "pronunciation_only",
                    "delta_norm": delta_for_synth.norm().item() if delta_for_synth is not None else 0.0,
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
