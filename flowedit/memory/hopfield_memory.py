"""
Modern Hopfield Memory — Stage 3 of FlowEdit (arXiv:2606.20518).

Paper Reference: Section 3.2 & Equations 5, 6, 7.
Continuous associative memory storing learned pronunciation corrections.

Mathematical Formulation:
    Key Storage (Eq. 5):
        K_i = pool(c_I) ∈ R^d
        V_i = pool(δ*_I) ∈ R^d
    Retrieval (Eq. 6):
        Mem(Q) = softmax(β Q K^T) V, where β = 1/√d
    Similarity Gating (Eq. 7):
        c_hat = c + σ(max_j(β Q K_j^T) - τ) ⊙ Mem(Q)
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowedit.config import MemoryConfig

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single pronunciation correction entry in Modern Hopfield Memory."""
    key: torch.Tensor           # K_i ∈ R^d (Gaussian context pooled base embedding)
    value: torch.Tensor         # V_i ∈ R^d (pooled latent perturbation δ*)
    word: str                   # Text representation of target word
    carrier_text: str           # Context sentence
    token_indices: List[int]    # Token indices I
    language: str = "en"
    access_count: int = 1
    last_access_step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Structured result from Modern Hopfield Memory retrieval."""
    retrieved_delta: torch.Tensor   # Mem(Q) ∈ R^[1, S, d] or R^[S, d]
    gate_values: torch.Tensor       # σ(max_j(β Q K_j^T) - τ) ∈ R^[1, S, 1]
    similarities: torch.Tensor      # Cosine similarity scores
    top_matches: List[Tuple[str, float]]
    is_active: bool


class HopfieldMemory(nn.Module):
    """Modern Continuous Hopfield Associative Memory for Lifelong Pronunciation Adaptation."""

    def __init__(self, config: Optional[MemoryConfig] = None, embedding_dim: int = 512):
        super().__init__()
        self.config = config or MemoryConfig()
        self.d = embedding_dim
        self.beta = self.config.hopfield_beta or (1.0 / math.sqrt(self.d))
        self.max_entries = getattr(self.config, "max_entries", 500)
        self.dedup_threshold = getattr(self.config, "dedup_cosine_threshold", 0.95)
        self.dedup_ema = getattr(self.config, "dedup_ema_decay", 0.90)
        self.context_window = getattr(self.config, "context_window", 3)
        self.context_sigma = getattr(self.config, "context_sigma", 1.5)

        # Gate threshold τ (Paper Section 3.2: τ ≈ 5.0)
        self.gate_threshold = nn.Parameter(
            torch.tensor(float(getattr(self.config, "gate_threshold_init", 5.0)), dtype=torch.float32)
        )

        self.entries: List[MemoryEntry] = []
        self.current_step = 0

    @property
    def num_entries(self) -> int:
        return len(self.entries)

    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def compute_context_key(
        self,
        embeddings: torch.Tensor,
        token_indices: List[int],
    ) -> torch.Tensor:
        """Compute Gaussian context-weighted key K_i ∈ R^d (Paper Section 3.2).

        K_i incorporates surrounding context window ±3 tokens:
            w_k = exp(-k^2 / (2 * σ^2)),  σ = 1.5
        """
        if embeddings.dim() == 3:
            emb = embeddings[0]  # [S, d]
        else:
            emb = embeddings     # [S, d]

        seq_len, dim = emb.shape

        if not token_indices:
            return F.normalize(emb.mean(dim=0), p=2, dim=-1)

        # Target span mean
        target_center = sum(token_indices) / len(token_indices)
        weights = torch.zeros(seq_len, device=emb.device, dtype=emb.dtype)

        for i in range(seq_len):
            if i in token_indices:
                weights[i] = 1.0
            else:
                dist = abs(i - target_center)
                if dist <= self.context_window:
                    weights[i] = math.exp(-(dist ** 2) / (2.0 * (self.context_sigma ** 2)))

        weights = weights / (weights.sum() + 1e-8)
        context_key = torch.sum(emb * weights.unsqueeze(-1), dim=0) # [d]
        return F.normalize(context_key, p=2, dim=-1)

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        word: str,
        carrier_text: str = "",
        token_indices: Optional[List[int]] = None,
        language: str = "en",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, str]:
        """Store or update a correction in Modern Hopfield Memory (Paper Section 3.2).

        If cosine similarity with existing entry > 0.95, perform EMA update:
            K_i ← α K_i + (1-α) K_new
            V_i ← α V_i + (1-α) V_new  (α = 0.90)
        """
        self.current_step += 1
        key_1d = key.squeeze().detach().float()
        val_1d = value.squeeze().detach().float()

        if key_1d.dim() > 1:
            key_1d = key_1d.mean(dim=0)
        if val_1d.dim() > 1:
            val_1d = val_1d.mean(dim=0)

        key_norm = F.normalize(key_1d, p=2, dim=-1)

        # Check for deduplication
        if self.entries:
            keys_stacked = torch.stack([F.normalize(e.key, p=2, dim=-1) for e in self.entries]).to(key_norm.device)
            sims = torch.mv(keys_stacked, key_norm)
            max_sim, best_idx = torch.max(sims, dim=0)

            if max_sim.item() >= self.dedup_threshold:
                # EMA Merge (Paper Section 3.2: α = 0.90)
                matched = self.entries[best_idx.item()]
                matched.key = F.normalize(self.dedup_ema * matched.key + (1.0 - self.dedup_ema) * key_1d, p=2, dim=-1)
                matched.value = self.dedup_ema * matched.value + (1.0 - self.dedup_ema) * val_1d
                matched.access_count += 1
                matched.last_access_step = self.current_step
                matched.metadata.update(metadata or {})
                logger.info(
                    f"✓ Hopfield Memory EMA Merge: '{word}' matched entry {best_idx} (sim={max_sim.item():.4f} ≥ {self.dedup_threshold})"
                )
                return best_idx.item(), "merged"

        # LRU Pruning if capacity exceeded (Paper Section 3.2: M_max = 500)
        if len(self.entries) >= self.max_entries:
            lru_idx = min(range(len(self.entries)), key=lambda i: self.entries[i].last_access_step)
            evicted = self.entries.pop(lru_idx)
            logger.info(f"Hopfield Memory Capacity Reached ({self.max_entries}). Evicted LRU entry '{evicted.word}'")

        entry = MemoryEntry(
            key=key_norm,
            value=val_1d,
            word=word,
            carrier_text=carrier_text,
            token_indices=token_indices or [],
            language=language,
            access_count=1,
            last_access_step=self.current_step,
            metadata=metadata or {},
        )
        self.entries.append(entry)
        idx = len(self.entries) - 1
        logger.info(f"✓ Hopfield Memory Stored: '{word}' at index {idx} (Total Entries: {len(self.entries)})")
        return idx, "inserted"

    def retrieve(
        self,
        query_embeddings: torch.Tensor,
        target_token_indices: Optional[List[int]] = None,
    ) -> RetrievalResult:
        """Stage 3: Retrieve corrections using Modern Hopfield Network (Paper Eq. 6 & 7).

        Mem(Q) = softmax(β Q K^T) V,  where β = 1/√d
        Gate = σ(max_j(β Q K_j^T) - τ)
        """
        device = query_embeddings.device
        if query_embeddings.dim() == 3:
            Q = query_embeddings[0]  # [S, d]
        else:
            Q = query_embeddings     # [S, d]

        seq_len, dim = Q.shape

        if self.is_empty():
            zeros_delta = torch.zeros_like(query_embeddings)
            zeros_gate = torch.zeros(1, seq_len, 1, device=device)
            return RetrievalResult(
                retrieved_delta=zeros_delta,
                gate_values=zeros_gate,
                similarities=torch.zeros(seq_len, 0, device=device),
                top_matches=[],
                is_active=False,
            )

        # 1. Stack memory keys K ∈ R^[M, d] and values V ∈ R^[M, d]
        K = torch.stack([e.key.to(device) for e in self.entries])       # [M, d]
        V = torch.stack([e.value.to(device) for e in self.entries])     # [M, d]

        # 2. Normalize Q and K for cosine dot-product
        Q_norm = F.normalize(Q, p=2, dim=-1)    # [S, d]
        K_norm = F.normalize(K, p=2, dim=-1)    # [M, d]

        # 3. Hopfield Attention: Scores S_ij = β Q_i K_j^T (Paper Eq. 6)
        scores = self.beta * torch.matmul(Q_norm, K_norm.transpose(0, 1))  # [S, M]
        attention_weights = F.softmax(scores, dim=-1)                     # [S, M]

        # 4. Associative Value Retrieval: Mem(Q) = softmax(β Q K^T) V (Paper Eq. 6)
        retrieved_V = torch.matmul(attention_weights, V)                   # [S, d]

        # 5. Similarity Gating (Paper Section 3.2 & Eq. 7):
        # max_sim = max_j (β Q_i K_j^T)
        # gate_i = σ(max_sim_i - τ)
        max_scores, max_indices = torch.max(scores, dim=-1)                # [S]
        tau = self.gate_threshold.to(device)
        gate = torch.sigmoid(max_scores - tau)                             # [S]

        # If target token indices are specified, focus retrieval on target tokens
        if target_token_indices:
            mask = torch.zeros(seq_len, 1, device=device)
            for idx in target_token_indices:
                if 0 <= idx < seq_len:
                    mask[idx] = 1.0
            gate = gate.unsqueeze(-1) * mask
        else:
            gate = gate.unsqueeze(-1)                                      # [S, 1]

        retrieved_delta = (gate * retrieved_V).unsqueeze(0)                # [1, S, d]
        gate_out = gate.unsqueeze(0)                                       # [1, S, 1]

        # Top matches for logging/inspection
        top_matches = []
        best_token_idx = torch.argmax(max_scores).item()
        best_entry_idx = max_indices[best_token_idx].item()
        best_score = max_scores[best_token_idx].item()
        if best_entry_idx < len(self.entries):
            top_matches.append((self.entries[best_entry_idx].word, best_score))

        is_active = (gate.max().item() > 0.05)

        return RetrievalResult(
            retrieved_delta=retrieved_delta,
            gate_values=gate_out,
            similarities=scores,
            top_matches=top_matches,
            is_active=is_active,
        )

    def clear(self) -> None:
        """Clear all stored memory entries."""
        self.entries.clear()
        self.current_step = 0
        logger.info("Hopfield Memory cleared.")
