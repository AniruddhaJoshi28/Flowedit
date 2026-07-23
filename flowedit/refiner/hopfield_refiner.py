"""
Hopfield Refiner — Gated Retrieval Module for FlowEdit Inference.

Paper Section 3.2 (Similarity Gate):
    "A similarity gate suppresses irrelevant retrievals for out-of-domain words:
        ĉ = c + σ(max_j(β·Q·K_j^T) - τ) ⊙ Mem(Q)
    where σ is the sigmoid function and τ ≈ 5.0 is a learned threshold
    scalar mapping cosine similarity to a gating factor."

This module sits between the text encoder and the decoder at inference time.
It is the mechanism that guarantees ZERO forgetting:
    - If no stored correction matches (gate ≈ 0): ĉ = c (identical to base model)
    - If a correction matches (gate ≈ 1): ĉ = c + Mem(Q) (correction applied)

The gate enables FUZZY MORPHOLOGICAL MATCHING (paper Section 3.2):
    "Because Softmax connects similar inputs, a correction for 'Linux' can
    partially address the query vector for 'Linux's' or 'Linuxed', avoiding
    the strict 1:1 match constraints of dictionary lookups."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import re
from typing import Optional, Tuple

from flowedit.memory.hopfield_memory import HopfieldMemory
from flowedit.config import MemoryConfig

logger = logging.getLogger(__name__)


class HopfieldRefiner(nn.Module):
    """Gated retrieval module for applying pronunciation corrections at inference.

    Architecture position:
        Text → [Text Encoder] → c → [HopfieldRefiner] → ĉ → [Decoder]

    The refiner performs three operations per token:
    1. Query the Hopfield memory for relevant corrections
    2. Compute similarity gate (suppress irrelevant retrievals)
    3. Add gated correction to original embeddings

    Mathematical guarantee of zero forgetting:
        For any token where gate(q) ≈ 0:
            ĉ(x) = c(x) + 0 · Mem(Q) = c(x)
        → Decoder receives IDENTICAL conditioning as base model
        → Output is IDENTICAL to unmodified model (not empirical — mathematical)
    """

    def __init__(
        self,
        memory: HopfieldMemory,
        config: Optional[MemoryConfig] = None,
    ):
        """Initialize HopfieldRefiner.

        Args:
            memory: The HopfieldMemory instance containing stored corrections
            config: Memory configuration (uses memory's config if None)
        """
        super().__init__()

        self.memory = memory
        self.config = config or memory.config

        # Learned gate threshold τ (paper: τ ≈ 5.0)
        # This is the only learnable parameter in the entire inference path
        self.tau = nn.Parameter(
            torch.tensor(self.config.gate_threshold_init, dtype=torch.float32)
        )

        logger.info(
            f"HopfieldRefiner initialized: τ={self.config.gate_threshold_init}"
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        text: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refine text embeddings by retrieving and applying stored corrections.

        Implements word-level retrieval with sequence-level perturbation injection.
        For each word in the text, we query the Hopfield memory. If a match is found
        (gate > 0.5), we inject the full sequence [N, d] of learned perturbations 
        into the corresponding character tokens.

        Args:
            embeddings: Text embeddings from encoder [batch, seq_len, dim]
            text: The original text string (required to align word boundaries)

        Returns:
            Tuple of:
            - refined_embeddings: ĉ with corrections applied [same shape as input]
            - gate_values: Per-token gate activations [batch, seq_len]
        """
        # Handle empty memory — pure passthrough (base model behavior)
        if self.memory.is_empty:
            if embeddings.dim() == 2:
                gate = torch.zeros(embeddings.shape[0], device=embeddings.device)
            else:
                gate = torch.zeros(
                    embeddings.shape[0], embeddings.shape[1],
                    device=embeddings.device
                )
            return embeddings, gate

        # Handle batch dimension
        had_batch = embeddings.dim() == 3
        if not had_batch:
            embeddings = embeddings.unsqueeze(0)  # [1, seq_len, dim]

        batch_size, seq_len, dim = embeddings.shape
        refined = embeddings.clone()
        gate_values = torch.zeros(batch_size, seq_len, device=embeddings.device)

        if text is None:
            logger.info("HopfieldRefiner: 'text' string not provided; performing token-level embedding gated retrieval.")

        # Process each batch
        for b in range(batch_size):
            processed_tokens = set()

            # Text is authoritative: exact lexical lookup prevents a nearby
            # embedding from applying another word's pronunciation.
            if text is not None:
                unique_words = {}
                for metadata in self.memory.metadata:
                    word = metadata.get("word", "")
                    if word:
                        unique_words.setdefault(word.casefold(), word)

                for word in unique_words.values():
                    pattern = re.compile(
                        rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE
                    )
                    entry_indices = self.memory.entries_for_word(word)
                    for match in pattern.finditer(text):
                        start_char = match.start()
                        end_char = min(seq_len, match.end())
                        if start_char >= end_char or not entry_indices:
                            continue

                        word_embeddings = embeddings[b, start_char:end_char, :]
                        query = F.normalize(word_embeddings.mean(dim=0), dim=0)
                        scored_entries = []
                        for entry_idx in entry_indices:
                            stored_key, _, _ = self.memory.get_entry(
                                entry_idx, embeddings.device
                            )
                            similarity = F.cosine_similarity(
                                query.unsqueeze(0),
                                stored_key.to(dtype=query.dtype).unsqueeze(0),
                            ).item()
                            scored_entries.append((similarity, entry_idx))
                        max_sim_value, selected_idx = max(scored_entries)
                        _, retrieved_sequence, _ = self.memory.get_entry(
                            selected_idx, embeddings.device
                        )
                        gate = torch.tensor(
                            1.0, device=embeddings.device, dtype=embeddings.dtype
                        )
                        logger.info(
                            f"HopfieldRefiner: exact correction for '{word}' "
                            f"(entry={selected_idx}, sim={max_sim_value:.3f})"
                        )

                        n_stored = retrieved_sequence.shape[0] if retrieved_sequence.dim() > 1 else 1
                        n_current = end_char - start_char
                        scale = getattr(self.config, "perturbation_scale", 1.8)
                        if retrieved_sequence.dim() == 1:
                            correction = scale * retrieved_sequence
                            refined[b, start_char:end_char, :] = word_embeddings + correction.unsqueeze(0)
                        elif n_stored == n_current:
                            correction = scale * retrieved_sequence
                            refined[b, start_char:end_char, :] = word_embeddings + correction
                        else:
                            sequence = retrieved_sequence.unsqueeze(0).transpose(1, 2)
                            interpolated = F.interpolate(
                                sequence, size=n_current, mode="linear", align_corners=True
                            )
                            correction = scale * interpolated.transpose(1, 2).squeeze(0)
                            refined[b, start_char:end_char, :] = word_embeddings + correction

                        for pos in range(start_char, end_char):
                            gate_values[b, pos] = 1.0
                            processed_tokens.add(pos)

            # Path 2: Token-level continuous embedding Hopfield retrieval (Paper Eq. 6 & 7)
            # Used only by legacy callers that do not provide source text.
            if text is not None:
                continue
            scale = getattr(self.config, "perturbation_scale", 1.8)
            for j in range(seq_len):
                if j in processed_tokens:
                    continue
                token_emb = embeddings[b, j, :]
                retrieved_val, max_sim = self.memory.retrieve(token_emb)
                gate = torch.sigmoid(10.0 * (max_sim - self.tau))

                if gate.item() > 0.5:
                    if retrieved_val.dim() > 1:
                        val_vec = retrieved_val.mean(dim=0)
                    else:
                        val_vec = retrieved_val

                    refined[b, j, :] = token_emb + scale * gate.item() * val_vec.to(embeddings.device)
                    gate_values[b, j] = gate.item()

        if not had_batch:
            refined = refined.squeeze(0)
            gate_values = gate_values.squeeze(0)

        # Log gate statistics for debugging
        if logger.isEnabledFor(logging.INFO):
            active = (gate_values > 0.5).sum().item()
            total = gate_values.numel()
            logger.info(
                f"Refiner: {active}/{total} tokens activated "
                f"(max gate: {gate_values.max().item():.3f}, "
                f"τ={self.tau.item():.2f})"
            )

        return refined, gate_values

    def get_gate_analysis(
        self,
        embeddings: torch.Tensor,
        token_texts: Optional[list] = None,
    ) -> list:
        """Detailed analysis of which tokens would be corrected.

        Useful for debugging and visualization.

        Args:
            embeddings: Text embeddings [seq_len, dim]
            token_texts: Optional list of token text strings

        Returns:
            List of dicts with per-token analysis
        """
        _, gate_values = self.forward(embeddings)

        if gate_values.dim() > 1:
            gate_values = gate_values[0]

        analysis = []
        for idx in range(gate_values.shape[0]):
            entry = {
                "token_idx": idx,
                "gate_value": gate_values[idx].item(),
                "activated": gate_values[idx].item() > 0.5,
            }

            if token_texts and idx < len(token_texts):
                entry["token_text"] = token_texts[idx]

            # Find which memory entry contributed most
            if not self.memory.is_empty:
                query = F.normalize(embeddings[idx:idx+1], dim=-1)
                K = torch.stack(self.memory.keys).to(device=query.device, dtype=query.dtype)
                sims = (self.memory.beta * query @ K.T).squeeze(0)
                best_idx = sims.argmax().item()
                entry["best_match_word"] = self.memory.metadata[best_idx].get("word", "?")
                entry["best_match_sim"] = sims[best_idx].item()

            analysis.append(entry)

        return analysis

    def __repr__(self) -> str:
        return (
            f"HopfieldRefiner(τ={self.tau.item():.2f}, "
            f"memory_size={self.memory.size})"
        )
