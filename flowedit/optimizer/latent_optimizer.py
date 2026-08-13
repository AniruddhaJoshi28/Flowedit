"""
Latent Optimizer — Stage 2 of FlowEdit (arXiv:2606.20518).

Paper Section 3.1 (Stage 2: Latent Input Optimization):
    "δ* = argmin_δ [ ||Mel(g_θ(c + δ)) - Mel(y_ref)||_2^2 + λ||δ||_2^2 ]
    where c = E(x) denotes text-encoder embeddings, g_θ denotes synthesis
    through frozen DiT and ODE solver, and λ = 0.001 is regularization weight.
    We mask non-target positions (δ_j = 0 ∀ j ∉ I)."

Optimizer Setup (Paper Section 4.1):
    - Optimizer: Adam
    - Steps: 50
    - Learning rate: Cosine annealing from η0 = 0.01 to η50 = 0.001
    - Gradient clipping: ||∇δ||_∞ ≤ 1.0
    - Masking: Target tokens + 1 token boundary expansion on each side
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from flowedit.config import OptimizationConfig

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Structured result from Stage 2 latent optimization."""
    delta: torch.Tensor             # Target token perturbation δ* [1, S, d]
    delta_pooled: torch.Tensor      # Pooled perturbation V_i ∈ R^d
    target_token_indices: List[int] # Target token indices I
    initial_loss: float
    final_loss: float
    total_steps: int
    delta_norm: float
    relative_delta_ratio: float
    converged: bool
    grad_history: List[float]


class LatentOptimizer:
    """Differentiable Latent Optimizer for FlowEdit."""

    def __init__(self, config: Optional[OptimizationConfig] = None, audio_config: Optional[Any] = None):
        self.config = config or OptimizationConfig()
        self.audio_config = audio_config

    def optimize(
        self,
        backbone,
        text: str,
        target_word: str,
        ref_audio_path: str,
        token_indices: List[int],
        speaker_conditioning: Dict[str, Any],
        language: str = "en",
        target_word_start_sample: Optional[int] = None,
        target_word_end_sample: Optional[int] = None,
        seed: int = 42,
    ) -> OptimizationResult:
        """Run 50-step Adam latent optimization with projected relative norm constraints.

        Args:
            backbone: F5TTSBackbone instance
            text: Full carrier sentence
            target_word: Word being corrected
            ref_audio_path: Reference audio containing target pronunciation
            token_indices: Base target token indices from alignment
            speaker_conditioning: Speaker embedding dict
            language: Language code
            target_word_start_sample: Sample start in baseline synthesis
            target_word_end_sample: Sample end in baseline synthesis
            seed: Reproducible seed for z0 ~ p0

        Returns:
            OptimizationResult containing delta* and diagnostic metadata
        """
        # Set reproducible seed for z0 ~ p0 ODE initialization
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Step 1: Encode baseline text embeddings c
        base_embeddings = backbone.encode_text(text, language) # [1, S, d]
        seq_len = base_embeddings.shape[1]
        embed_dim = base_embeddings.shape[2]

        # Step 2: Expand target token indices by ±1 token to absorb tokenizer boundary errors (Paper Section 3.1)
        expanded_indices = set()
        for idx in token_indices:
            expanded_indices.add(idx)
            if idx > 0:
                expanded_indices.add(idx - 1)
            if idx < seq_len - 1:
                expanded_indices.add(idx + 1)
        target_indices = sorted(list(expanded_indices))

        if not target_indices:
            target_indices = list(range(seq_len))

        # Step 3: Create target token mask M ∈ {0, 1}^S
        mask = torch.zeros_like(base_embeddings, device=base_embeddings.device)
        for idx in target_indices:
            mask[:, idx, :] = 1.0

        # Step 4: Initialize raw delta parameter
        delta = torch.zeros_like(base_embeddings, device=base_embeddings.device, requires_grad=True)

        # Hyperparameters (Paper Section 4.1)
        n_steps = getattr(self.config, "n_steps", 50)
        lr_init = getattr(self.config, "lr_start", 0.005)
        lr_final = getattr(self.config, "lr_end", 0.0005)
        lambda_reg = getattr(self.config, "lambda_reg", 0.001)
        max_grad_norm = getattr(self.config, "grad_clip_max_norm", 1.0)
        alpha_max = getattr(self.config, "max_relative_delta", 0.15)

        optimizer = torch.optim.Adam([delta], lr=lr_init)

        # Baseline text embedding norm over target tokens
        base_target_norm = torch.norm(base_embeddings[:, target_indices, :]).item() + 1e-8

        initial_loss = 0.0
        final_loss = 0.0
        grad_history = []
        nan_count = 0

        logger.info(
            f"[FlowEdit Optimization] Target: '{target_word}', Tokens: {target_indices}, "
            f"base_target_norm={base_target_norm:.4f}, steps={n_steps}"
        )

        for step in range(n_steps):
            optimizer.zero_grad()

            # Cosine LR decay: η0 = 0.01 -> η50 = 0.001
            progress = step / max(1, n_steps - 1)
            current_lr = lr_final + 0.5 * (lr_init - lr_final) * (1.0 + math.cos(math.pi * progress))
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Apply target mask: δ_M = M ⊙ δ
            delta_masked = delta * mask

            # Forward pass: Differentiable ODE integration over c + δ_M
            loss_dict = backbone.compute_optimization_loss(
                text_embedding_delta=delta_masked,
                ref_audio_path=ref_audio_path,
                speaker_conditioning=speaker_conditioning,
                text=text,
                language=language,
                target_word_start_sample=target_word_start_sample,
                target_word_end_sample=target_word_end_sample,
                seed=seed,
            )

            mel_loss = loss_dict["loss"]
            reg_loss = lambda_reg * torch.sum(delta_masked ** 2)
            total_loss = mel_loss + reg_loss

            if step == 0:
                initial_loss = total_loss.item()
            final_loss = total_loss.item()

            if math.isnan(final_loss):
                nan_count += 1
                if nan_count >= 3:
                    logger.warning(f"Optimization diverged to NaN for 3 consecutive steps. Aborting early at step {step}.")
                    break
            else:
                nan_count = 0

            total_loss.backward()

            # Verify gradient health & sanitize non-finite elements
            assert delta.requires_grad, "delta must have requires_grad=True"
            assert delta.grad is not None, f"Step {step}: delta.grad is None! Autograd chain broken."
            
            # Clean any numerical overflow/underflow spikes in mixed precision backward pass
            torch.nan_to_num_(delta.grad, nan=0.0, posinf=max_grad_norm, neginf=-max_grad_norm)

            grad_norm = delta.grad.norm().item()
            grad_history.append(grad_norm)

            # L∞ gradient clipping ≤ 1.0 (Paper Section 4.1)
            torch.nn.utils.clip_grad_value_([delta], clip_value=max_grad_norm)

            optimizer.step()

            # Projected update for relative norm constraint: ||δ_I|| / ||c_I|| ≤ α
            with torch.no_grad():
                delta.mul_(mask)
                cur_delta_norm = torch.norm(delta[:, target_indices, :]).item()
                rel_ratio = cur_delta_norm / base_target_norm
                if rel_ratio > alpha_max:
                    scale = alpha_max / rel_ratio
                    delta.mul_(scale)

            # Log step metrics
            if step % 10 == 0 or step == n_steps - 1:
                cur_norm = torch.norm(delta[:, target_indices, :]).item()
                rel_ratio = cur_norm / base_target_norm
                logger.info(
                    f"[Step {step:02d}/{n_steps}] Total Loss: {total_loss.item():.4f} | "
                    f"Mel Loss: {mel_loss.item():.4f} | δ L2 Norm: {cur_norm:.4f} | "
                    f"Rel Ratio: {rel_ratio:.4f} | Grad Norm: {grad_norm:.4f} | LR: {current_lr:.6f}"
                )

        # Final candidate perturbation δ*
        with torch.no_grad():
            final_delta = (delta * mask).detach()
            final_norm = torch.norm(final_delta[:, target_indices, :]).item()
            final_rel_ratio = final_norm / base_target_norm

            # Pool target token perturbation to single vector V_i ∈ R^d (Paper Eq. 5: V_i = pool(δ*_I))
            if len(target_indices) > 0:
                pooled_value = final_delta[0, target_indices, :].mean(dim=0) # [d]
            else:
                pooled_value = final_delta[0].mean(dim=0) # [d]

        logger.info(
            f"✓ FlowEdit Optimization Complete: Initial Loss={initial_loss:.4f} → "
            f"Final Loss={final_loss:.4f}, δ* Norm={final_norm:.4f}, Rel Ratio={final_rel_ratio:.4f}"
        )

        return OptimizationResult(
            delta=final_delta,
            delta_pooled=pooled_value,
            target_token_indices=target_indices,
            initial_loss=initial_loss,
            final_loss=final_loss,
            total_steps=n_steps,
            delta_norm=final_norm,
            relative_delta_ratio=final_rel_ratio,
            converged=(final_loss < initial_loss),
            grad_history=grad_history,
        )
