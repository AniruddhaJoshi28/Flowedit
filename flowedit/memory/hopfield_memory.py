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

import os
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
    full_delta: Optional[torch.Tensor] = None  # Full δ* ∈ R^[1, S, d] (pre-ConvNeXt space, per-character)


@dataclass
class RetrievalResult:
    """Structured result from Modern Hopfield Memory retrieval."""
    retrieved_delta: torch.Tensor   # Mem(Q) ∈ R^[1, S, d] or R^[S, d]
    gate_values: torch.Tensor       # σ(max_j(β Q K_j^T) - τ) ∈ R^[1, S, 1]
    similarities: torch.Tensor      # Cosine similarity scores
    top_matches: List[Tuple[str, float]]
    is_active: bool
    matched_full_delta: Optional[torch.Tensor] = None  # Full δ* from best entry [1, S_orig, d]
    matched_word: Optional[str] = None                 # Word from best matching entry
    matched_token_indices: Optional[List[int]] = None   # Token indices from best entry
    matched_carrier_text: Optional[str] = None          # Original carrier text from best entry


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
        self.context_window = getattr(self.config, "context_window", 1)
        self.context_sigma = getattr(self.config, "context_sigma", 0.8)

        # Gate threshold τ (Paper Section 3.2: τ ≈ 9.0 for precise homograph gating)
        self.gate_threshold = nn.Parameter(
            torch.tensor(float(getattr(self.config, "gate_threshold_init", 9.0)), dtype=torch.float32)
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

        K_i incorporates surrounding context window ±1-2 tokens:
            w_k = exp(-k^2 / (2 * σ^2))
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

        char_radius = max(2, int(self.context_window * 2))
        char_sigma = max(0.5, float(self.context_sigma * 1.5))

        for i in range(seq_len):
            if i in token_indices:
                weights[i] = 1.0
            else:
                dist = abs(i - target_center)
                if dist <= char_radius:
                    weights[i] = math.exp(-(dist ** 2) / (2.0 * (char_sigma ** 2)))

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
        full_delta: Optional[torch.Tensor] = None,
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
                matched_key = matched.key.to(device=key_1d.device, dtype=key_1d.dtype)
                matched_val = matched.value.to(device=val_1d.device, dtype=val_1d.dtype)
                matched.key = F.normalize(self.dedup_ema * matched_key + (1.0 - self.dedup_ema) * key_1d, p=2, dim=-1)
                matched.value = self.dedup_ema * matched_val + (1.0 - self.dedup_ema) * val_1d
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

        # Store full_delta detached on CPU to save GPU memory
        stored_full_delta = full_delta.detach().cpu().float() if full_delta is not None else None
        if stored_full_delta is not None:
            logger.info(
                f"  Storing full δ* tensor: shape={list(stored_full_delta.shape)}, "
                f"norm={stored_full_delta.norm().item():.4f}, "
                f"non_zero_positions={int((stored_full_delta.abs().sum(dim=-1) > 1e-6).sum().item())}"
            )

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
            full_delta=stored_full_delta,
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

        Mem(Q) = softmax(β Q K^T) V,  where β is temperature scaling
        Gate = σ(max_j(β Q K_j^T) - τ)
        """
        device = query_embeddings.device
        dtype = query_embeddings.dtype
        if query_embeddings.dim() == 3:
            Q = query_embeddings[0]  # [S, d]
        else:
            Q = query_embeddings     # [S, d]

        seq_len, dim = Q.shape

        if self.is_empty():
            zeros_delta = torch.zeros_like(query_embeddings)
            zeros_gate = torch.zeros(1, seq_len, 1, device=device, dtype=dtype)
            return RetrievalResult(
                retrieved_delta=zeros_delta,
                gate_values=zeros_gate,
                similarities=torch.zeros(seq_len, 0, device=device, dtype=dtype),
                top_matches=[],
                is_active=False,
            )

        # 1. Stack memory keys K ∈ R^[M, d] and values V ∈ R^[M, d] with matching dtype & device
        K = torch.stack([e.key.to(device=device, dtype=dtype) for e in self.entries])       # [M, d]
        V = torch.stack([e.value.to(device=device, dtype=dtype) for e in self.entries])     # [M, d]

        # 2. Compute contextual query vectors Q_ctx across sequence using Gaussian smoothing
        char_radius = max(2, int(self.context_window * 2))
        char_sigma = max(0.5, float(self.context_sigma * 1.5))
        kernel_size = 2 * char_radius + 1
        x_grid = torch.arange(-char_radius, char_radius + 1, device=device, dtype=dtype)
        gaussian_kernel = torch.exp(- (x_grid ** 2) / (2.0 * (char_sigma ** 2)))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        Q_unsq = Q.unsqueeze(0).transpose(1, 2)  # [1, d, S]
        Q_padded = F.pad(Q_unsq, (char_radius, char_radius), mode='replicate')
        kernel_weight = gaussian_kernel.view(1, 1, kernel_size).repeat(dim, 1, 1)
        Q_ctx = F.conv1d(Q_padded, kernel_weight, groups=dim).transpose(1, 2).squeeze(0)  # [S, d]

        # 3. Normalize Q_ctx and K for cosine dot-product
        Q_norm = F.normalize(Q_ctx, p=2, dim=-1)  # [S, d]
        K_norm = F.normalize(K, p=2, dim=-1)      # [M, d]

        # 4. Hopfield Attention: Scores S_ij = β Q_i K_j^T (Paper Eq. 6)
        # Scaled inverse temperature β for normalized cosine similarity
        beta_scale = 10.0 if (self.beta < 1.0) else float(self.beta)
        scores = beta_scale * torch.matmul(Q_norm, K_norm.transpose(0, 1))  # [S, M]
        attention_weights = F.softmax(scores, dim=-1)                        # [S, M]

        # 5. Associative Value Retrieval: Mem(Q) = softmax(β Q K^T) V (Paper Eq. 6)
        retrieved_V = torch.matmul(attention_weights, V)                      # [S, d]

        # 6. Similarity Gating (Paper Section 3.2 & Eq. 7):
        # max_sim = max_j (β Q_i K_j^T)
        # gate_i = σ(max_sim_i - τ)
        max_scores, max_indices = torch.max(scores, dim=-1)                   # [S]
        tau = self.gate_threshold.to(device=device, dtype=dtype)
        gate = torch.sigmoid(max_scores - tau)                                # [S]

        # Suppress noise below threshold (gate > 0.25)
        gate_final = torch.where(gate > 0.25, gate, torch.zeros_like(gate))

        # Normalize active gate so peak reaches 1.0 (delivering full learned delta norm)
        max_g = gate_final.max()
        if max_g > 0.05:
            gate_final = gate_final / max_g

        active_idx = (gate_final > 0).nonzero(as_tuple=True)[0].tolist()
        logger.info(
            f"[Hopfield Retrieve] Post-threshold active positions: {active_idx} / seq_len={seq_len}, "
            f"values={[round(gate_final[i].item(), 3) for i in active_idx]}, "
            f"scores={[round(max_scores[i].item(), 3) for i in active_idx]}"
        )
        gate = gate_final

        # If target token indices are specified, focus retrieval on target tokens
        if target_token_indices:
            mask = torch.zeros(seq_len, 1, device=device, dtype=dtype)
            for idx in target_token_indices:
                if 0 <= idx < seq_len:
                    mask[idx] = 1.0
            gate = gate.unsqueeze(-1) * mask
        else:
            gate = gate.unsqueeze(-1)                                         # [S, 1]

        retrieved_delta = (gate * retrieved_V).unsqueeze(0)                   # [1, S, d]
        gate_out = gate.unsqueeze(0)                                          # [1, S, 1]

        # Top matches for diagnostics
        top_matches = []
        best_token_idx = torch.argmax(max_scores).item()
        best_entry_idx = max_indices[best_token_idx].item()
        best_score = max_scores[best_token_idx].item()

        matched_full_delta = None
        matched_word = None
        matched_token_indices = None
        matched_carrier_text = None

        if best_entry_idx < len(self.entries):
            best_entry = self.entries[best_entry_idx]
            top_matches.append((best_entry.word, best_score))
            matched_full_delta = best_entry.full_delta
            matched_word = best_entry.word
            matched_token_indices = best_entry.token_indices
            matched_carrier_text = best_entry.carrier_text
            if matched_full_delta is not None:
                logger.info(
                    f"[Hopfield Retrieve] Best match: '{best_entry.word}' (entry {best_entry_idx}, "
                    f"score={best_score:.3f}), full_delta norm={matched_full_delta.norm().item():.4f}"
                )

        is_active = (gate.max().item() > 0.05)

        return RetrievalResult(
            retrieved_delta=retrieved_delta,
            gate_values=gate_out,
            similarities=scores,
            top_matches=top_matches,
            is_active=is_active,
            matched_full_delta=matched_full_delta,
            matched_word=matched_word,
            matched_token_indices=matched_token_indices,
            matched_carrier_text=matched_carrier_text,
        )

    def save(self, filepath: str) -> None:
        """Serialize memory entries to disk."""
        data = {
            "entries": [
                {
                    "key": e.key.cpu(),
                    "value": e.value.cpu(),
                    "word": e.word,
                    "carrier_text": e.carrier_text,
                    "token_indices": e.token_indices,
                    "language": e.language,
                    "access_count": e.access_count,
                    "last_access_step": e.last_access_step,
                    "metadata": e.metadata,
                    "full_delta": e.full_delta.cpu() if e.full_delta is not None else None,
                }
                for e in self.entries
            ],
            "current_step": self.current_step,
            "gate_threshold": self.gate_threshold.data.cpu(),
        }
        torch.save(data, filepath)
        logger.info(f"✓ Saved {len(self.entries)} Hopfield Memory entries to {filepath}")

    def load(self, filepath: str) -> None:
        """Load serialized memory entries from disk."""
        if not os.path.exists(filepath):
            return
        data = torch.load(filepath, map_location="cpu", weights_only=False)
        self.entries = [
            MemoryEntry(
                key=item["key"],
                value=item["value"],
                word=item["word"],
                carrier_text=item.get("carrier_text", ""),
                token_indices=item.get("token_indices", []),
                language=item.get("language", "en"),
                access_count=item.get("access_count", 1),
                last_access_step=item.get("last_access_step", 0),
                metadata=item.get("metadata", {}),
                full_delta=item.get("full_delta", None),
            )
            for item in data.get("entries", [])
        ]
        self.current_step = data.get("current_step", 0)
        if "gate_threshold" in data:
            self.gate_threshold.data = data["gate_threshold"].to(self.gate_threshold.device)
        logger.info(f"✓ Loaded {len(self.entries)} Hopfield Memory entries from {filepath}")

    def delete_entry(self, word: str) -> bool:
        """Delete an entry or entries matching target word from memory.

        Args:
            word: Target word string to remove (matched case-insensitively).

        Returns:
            bool: True if one or more entries were deleted, False if not found.
        """
        initial_len = len(self.entries)
        target_norm = word.strip().lower()
        self.entries = [
            e for e in self.entries
            if e.word.strip().lower() != target_norm
        ]
        deleted = len(self.entries) < initial_len
        if deleted:
            logger.info(
                f"✓ Hopfield Memory: Deleted correction for '{word}' "
                f"({initial_len - len(self.entries)} removed, {len(self.entries)} remaining)"
            )
        else:
            logger.warning(f"Hopfield Memory: No entry found to delete for word '{word}'")
        return deleted

    def clear(self) -> None:
        """Clear all stored memory entries."""
        self.entries.clear()
        self.current_step = 0
        if os.path.exists("./corrections.pt"):
            try:
                os.remove("./corrections.pt")
            except Exception:
                pass
        logger.info("Hopfield Memory cleared.")

