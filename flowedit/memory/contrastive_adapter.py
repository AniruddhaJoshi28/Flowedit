"""
Contrastive Pronunciation Adapter (CPA) for SemFlow-Memory.

Projects contextual embeddings (e.g. MiniLM d=384 or DiT d=1024) into a compact,
pronunciation-discriminative subspace (d=256) where same-pronunciation contexts
cluster tightly and competing pronunciations of the same spelling are pushed apart.

Zero word-specific hardcoding: fully generalized for arbitrary words, sentences, and dimensions.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020 / InfoNCE).

    Pulls together embeddings of the same pronunciation sense while pushing
    apart embeddings of opposing senses and competing homograph pronunciations.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Args:

        features: [B, D] L2-normalized feature vectors. labels: [B] integer class
        labels (e.g., sense IDs).
        """
        device = features.device
        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Pairwise cosine similarity matrix: [B, B]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create mask for same-label pairs: [B, B]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Mask out self-contrast (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0.0,
        )
        mask = mask * logits_mask

        # Numerical stability: subtract max per row
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Compute log probabilities: exp(sim) / sum_a exp(sim)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Mean of log-likelihood over positive pairs
        positive_counts = mask.sum(1)
        # Avoid division by zero for isolated classes with count 0
        valid_indices = positive_counts > 0
        if not valid_indices.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (mask[valid_indices] * log_prob[valid_indices]).sum(
            1
        ) / positive_counts[valid_indices]
        loss = -mean_log_prob_pos.mean()
        return loss


class ContrastivePronunciationAdapter(nn.Module):
    """Lightweight 2-layer MLP projection head with residual connection and LayerNorm.

    Maps contextual representations into a metric space optimized for pronunciation separation.
    """

    def __init__(
        self,
        in_dim: int = 384,
        hidden_dim: int = 384,
        out_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

        # Optional linear projection for residual if dimensions differ
        self.res_proj = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        """Args:

        x: [..., in_dim] raw contextual embeddings. normalize: Whether to apply
        L2 normalization to output vectors.
        """
        residual = self.res_proj(x)
        out = self.net(x) + residual
        if normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out
