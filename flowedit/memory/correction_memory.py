"""
Correction Memory Modules for SemFlow-Memory.

Compares:
1. CosineKNNMemory: Standard baseline nearest-neighbor retrieval with threshold abstention.
2. InterferenceAwareHopfieldMemory: Continuous Modern Hopfield Associative Memory with
   softmax association, competition margin gating, sequential EMA prototype updating,
   and interference-aware retrieval abstention.

Completely generalized: no word-specific hardcoding. Operates on arbitrary words, spellings,
and phonetic representations.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MemoryEntry:
    """A stored memory entry representing a verified pronunciation correction."""
    key: torch.Tensor          # [D] normalized key vector
    value: torch.Tensor        # [D_v] target correction value / phonetic vector
    spelling: str              # Surface word spelling (e.g., 'lead', 'read', etc.)
    sense_id: int              # Pronunciation sense ID
    label: str                 # Human-readable phoneme label (e.g., '/lɛd/')
    sample_count: int = 1      # Number of observations merged into this prototype
    metadata: Dict[str, Any] = field(default_factory=dict)


class CosineKNNMemory:
    """Standard Cosine Nearest-Neighbor Memory Baseline.

    Retrieves the value of the single nearest stored key if cosine similarity
    exceeds an abstention threshold tau.
    """

    def __init__(self, abstain_threshold: float = 0.65):
        self.abstain_threshold = abstain_threshold
        self.entries: List[MemoryEntry] = []

    def add_entry(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        spelling: str,
        sense_id: int,
        label: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Stores a new correction entry."""
        k = F.normalize(key.detach().cpu().float().flatten(), p=2, dim=0)
        v = value.detach().cpu().float()
        entry = MemoryEntry(
            key=k,
            value=v,
            spelling=spelling.lower().strip(),
            sense_id=sense_id,
            label=label,
            metadata=metadata or {},
        )
        self.entries.append(entry)

    def retrieve(self, query: torch.Tensor, target_spelling: Optional[str] = None) -> Dict[str, Any]:
        """Queries the memory.

        Args:
            query: [D] query vector
            target_spelling: Optional filter to restrict candidates to a surface word
        Returns:
            Dict containing retrieved value, best similarity, best entry, and abstention status.
        """
        if not self.entries:
            return {"value": None, "similarity": 0.0, "is_abstained": True, "entry": None, "method": "knn"}

        q = F.normalize(query.detach().cpu().float().flatten(), p=2, dim=0)
        candidates = self.entries
        if target_spelling:
            target_norm = target_spelling.lower().strip()
            filtered = [e for e in self.entries if e.spelling == target_norm]
            if filtered:
                candidates = filtered

        best_sim = -1.0
        best_entry = None
        for entry in candidates:
            sim = float(torch.dot(q, entry.key).item())
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        is_abstained = best_sim < self.abstain_threshold
        retrieved_val = best_entry.value if (not is_abstained and best_entry is not None) else None

        return {
            "value": retrieved_val,
            "similarity": best_sim,
            "is_abstained": is_abstained,
            "entry": best_entry,
            "method": "knn",
        }

    def clear(self):
        self.entries.clear()


class InterferenceAwareHopfieldMemory:
    """Modern Continuous Hopfield Associative Memory with Interference & Margin Gating.

    Mathematical Formulation:
        1. Query-Key Association:
           a_i = softmax(beta * <q, k_i>)
        2. Associative Retrieval:
           v_retrieved = sum_i a_i * v_i
        3. Interference Detection:
           When multiple competing senses share the same surface spelling, measures
           the margin between the top-1 sense activation and top-2 competing sense activation:
               Delta_margin = a_(1) - a_(2)
        4. Gating / Abstention:
           Abstains (returns null / unperturbed baseline) if:
           - max similarity < tau_abstain (out-of-distribution or unseen word), OR
           - Delta_margin < theta_margin (high competition / interference between senses).
        5. Sequential Continual Updates:
           Repeated examples of the same sense are merged into an EMA prototype,
           preventing catastrophic forgetting and memory expansion blowup.
    """

    def __init__(
        self,
        beta: float = 8.0,
        abstain_threshold: float = 0.60,
        margin_threshold: float = 0.25,
        ema_alpha: float = 0.80,
    ):
        self.beta = beta
        self.abstain_threshold = abstain_threshold
        self.margin_threshold = margin_threshold
        self.ema_alpha = ema_alpha
        self.entries: List[MemoryEntry] = []

    def add_or_update_entry(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        spelling: str,
        sense_id: int,
        label: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Adds a new correction or updates an existing sense prototype via EMA."""
        k = F.normalize(key.detach().cpu().float().flatten(), p=2, dim=0)
        v = value.detach().cpu().float()
        spelling_norm = spelling.lower().strip()

        # Check for existing prototype of the exact same word and sense
        for existing in self.entries:
            if existing.spelling == spelling_norm and existing.sense_id == sense_id:
                # Merge into existing prototype via EMA
                merged_k = self.ema_alpha * existing.key + (1.0 - self.ema_alpha) * k
                existing.key = F.normalize(merged_k, p=2, dim=0)
                existing.value = self.ema_alpha * existing.value + (1.0 - self.ema_alpha) * v
                existing.sample_count += 1
                if metadata:
                    existing.metadata.update(metadata)
                return

        # New sense entry
        entry = MemoryEntry(
            key=k,
            value=v,
            spelling=spelling_norm,
            sense_id=sense_id,
            label=label,
            sample_count=1,
            metadata=metadata or {},
        )
        self.entries.append(entry)

    def retrieve(
        self,
        query: torch.Tensor,
        target_spelling: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Performs interference-aware associative memory retrieval.

        Args:
            query: [D] query vector
            target_spelling: Optional target word spelling
        Returns:
            Dict containing retrieved value, attention weights, interference margin,
            similarity, and abstention verdict.
        """
        if not self.entries:
            return {
                "value": None,
                "similarity": 0.0,
                "margin": 0.0,
                "is_abstained": True,
                "abstain_reason": "empty_memory",
                "entry": None,
                "weights": [],
                "method": "hopfield",
            }

        q = F.normalize(query.detach().cpu().float().flatten(), p=2, dim=0)
        candidates = self.entries
        if target_spelling:
            target_norm = target_spelling.lower().strip()
            filtered = [e for e in self.entries if e.spelling == target_norm]
            if filtered:
                candidates = filtered

        # 1. Compute cosine similarities
        sims = torch.tensor([torch.dot(q, e.key).item() for e in candidates], dtype=torch.float32)
        max_sim = float(sims.max().item())

        # 2. Modern Hopfield softmax association: a = softmax(beta * sims)
        attn_weights = F.softmax(self.beta * sims, dim=0)

        # 3. Compute interference margin between top-1 and top-2 competing entries
        if len(candidates) > 1:
            sorted_weights, sorted_indices = torch.sort(attn_weights, descending=True)
            top1_w = float(sorted_weights[0].item())
            top2_w = float(sorted_weights[1].item())
            margin = top1_w - top2_w
            best_idx = int(sorted_indices[0].item())
        else:
            margin = 1.0
            best_idx = 0

        best_entry = candidates[best_idx]

        # 4. Associative readout: v_out = sum_i a_i * v_i
        v_out = torch.zeros_like(candidates[0].value)
        for i, entry in enumerate(candidates):
            v_out = v_out + attn_weights[i] * entry.value

        # 5. Abstention evaluation
        is_abstained = False
        abstain_reason = "none"

        if max_sim < self.abstain_threshold:
            is_abstained = True
            abstain_reason = f"low_similarity (max_sim={max_sim:.3f} < tau={self.abstain_threshold})"
        elif margin < self.margin_threshold:
            is_abstained = True
            abstain_reason = f"interference_detected (margin={margin:.3f} < theta={self.margin_threshold})"

        return {
            "value": v_out if not is_abstained else None,
            "similarity": max_sim,
            "margin": margin,
            "is_abstained": is_abstained,
            "abstain_reason": abstain_reason,
            "entry": best_entry,
            "weights": [float(w.item()) for w in attn_weights],
            "method": "hopfield",
        }

    def clear(self):
        self.entries.clear()
