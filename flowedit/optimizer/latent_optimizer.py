"""
Latent Optimizer — Stage 2 of FlowEdit (arXiv:2606.20518).

Paper Section 3.2 (Stage 2: Latent Input Optimization):
    "δ* = argmin_δ [ ||Mel(g_θ(c + δ)) - Mel(y_ref)||_2^2 + λ||δ||_2^2 ]
    where c = E(x) denotes text-encoder embeddings, g_θ denotes synthesis
    through the frozen DiT and ODE solver (Eq. 2), and λ = 0.001 is a
    regularization weight. We mask non-target positions (δ_j = 0 ∀ j ∉ I)."

Enhanced for:
    - Multi-speaker joint optimization (learning speaker-invariant phonetic perturbations).
    - Canonical per-character word_delta extraction for Modern Hopfield Memory storage.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowedit.config import OptimizationConfig

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Structured result from Stage 2 latent optimization."""
    delta: torch.Tensor             # Target token perturbation δ* [1, S, d]
    delta_pooled: torch.Tensor      # Pooled perturbation V_i = pool(δ*_I) ∈ R^d
    key_pooled: torch.Tensor        # Pooled base key K_i = pool(c_I) ∈ R^d
    target_token_indices: List[int] # Target token indices I
    initial_loss: float
    final_loss: float
    total_steps: int
    delta_norm: float
    relative_delta_ratio: float
    converged: bool
    grad_history: List[float]
    base_embeddings: Optional[torch.Tensor] = None
    word_delta: Optional[torch.Tensor] = None   # Canonical word-span perturbation [1, L_word, d]


class LatentOptimizer:
    """Stage 2 Latent Input Optimizer for FlowEdit."""

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
        speaker_conditioning: Union[Dict[str, Any], List[Dict[str, Any]]],
        language: str = "en",
        target_word_start_sample: Optional[int] = None,
        target_word_end_sample: Optional[int] = None,
        ref_start_time: Optional[float] = None,
        ref_end_time: Optional[float] = None,
        phonetic_hint: Optional[str] = None,
        seed: int = 42,
    ) -> OptimizationResult:
        """Run 50-step Adam latent optimization (Paper Section 3.2 & 4.1).

        Args:
            backbone: TTSBackbone instance (XTTSBackbone or F5TTSBackbone)
            text: Full carrier sentence
            target_word: Word being corrected
            ref_audio_path: Reference audio containing target pronunciation
            token_indices: Base target token indices from alignment
            speaker_conditioning: Speaker embedding dict or list of dicts for multi-speaker joint optimization
            language: Language code
            target_word_start_sample: Sample start in baseline synthesis
            target_word_end_sample: Sample end in baseline synthesis
            ref_start_time: Time start of word in reference audio
            ref_end_time: Time end of word in reference audio
            seed: Reproducible seed for z0 ~ p0 ODE initialization

        Returns:
            OptimizationResult containing δ*, K_i, V_i, word_delta, and convergence diagnostics
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.empty_cache()

        # Step 1: Encode baseline text embeddings c = E(x) ∈ R^[1, S, d]
        base_embeddings = backbone.encode_text(text, language)  # [1, S, d]
        seq_len = base_embeddings.shape[1]
        embed_dim = base_embeddings.shape[2]

        # Step 2: Use aligned target token indices
        target_indices = sorted(list(token_indices)) if token_indices else list(range(seq_len))

        # Step 3: Create target token mask M ∈ {0, 1}^[1, S, d]
        mask = torch.zeros_like(base_embeddings, device=base_embeddings.device)
        for idx in target_indices:
            mask[:, idx, :] = 1.0

        # Step 4: Initialize learnable perturbation δ = 0 (Paper Section 3.2: "δ ∈ R^(S×d) initialized at zero")
        delta = torch.zeros(base_embeddings.shape, device=base_embeddings.device, dtype=torch.float32, requires_grad=True)

        # Normalize speaker_conditioning into a list for joint optimization
        if isinstance(speaker_conditioning, list):
            speaker_cond_list = speaker_conditioning
        else:
            speaker_cond_list = [speaker_conditioning]

        # Hyperparameters strictly from Paper Section 3.2 & 4.1
        n_steps = getattr(self.config, "n_steps", 100)
        lr_init = getattr(self.config, "lr_start", 0.080)
        lr_final = getattr(self.config, "lr_end", 0.002)
        lambda_reg = getattr(self.config, "lambda_reg", 0.0001)
        max_grad_norm = getattr(self.config, "grad_clip_max_norm", 1.0)
        ode_steps = getattr(self.config, "ode_steps", 16)
        alpha_max = float(getattr(self.config, "max_relative_delta", 1.20))

        base_target_norm = torch.norm(base_embeddings[:, target_indices, :]).item() if len(target_indices) > 0 else torch.norm(base_embeddings).item()
        base_target_norm = max(base_target_norm, 1e-4)
        max_delta_norm = alpha_max * base_target_norm

        optimizer = torch.optim.Adam([delta], lr=lr_init)

        initial_loss = 0.0
        final_loss = 0.0
        grad_history = []

        logger.info(
            f"[FlowEdit Optimization] Target: '{target_word}', Token Span I: {target_indices}, "
            f"Speakers: {len(speaker_cond_list)}, ||c_I||={base_target_norm:.4f}, Max ||δ||={max_delta_norm:.4f}, Steps={n_steps}, λ={lambda_reg}"
        )

        for step in range(n_steps):
            optimizer.zero_grad()

            # Cosine annealing learning rate schedule: η0 = 0.015 → η50 = 0.001 (Paper Section 3.2)
            progress = step / max(1, n_steps - 1)
            current_lr = lr_final + 0.5 * (lr_init - lr_final) * (1.0 + math.cos(math.pi * progress))
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Mask non-target positions: δ_M = M ⊙ δ (δ_j = 0 ∀ j ∉ I)
            delta_masked = delta * mask

            # Forward pass through frozen DiT and ODE solver (Eq. 2 & 3)
            speaker_mel_losses = []
            for s_cond in speaker_cond_list:
                loss_dict = backbone.compute_optimization_loss(
                    text_embedding_delta=delta_masked,
                    ref_audio_path=ref_audio_path,
                    speaker_conditioning=s_cond,
                    text=text,
                    target_word=target_word,
                    language=language,
                    target_word_start_sample=target_word_start_sample,
                    target_word_end_sample=target_word_end_sample,
                    ref_start_time=ref_start_time,
                    ref_end_time=ref_end_time,
                    seed=seed,
                    ode_steps=ode_steps,
                    token_indices=target_indices,
                )
                speaker_mel_losses.append(loss_dict["loss"])

            mel_loss = torch.stack(speaker_mel_losses).mean() if len(speaker_mel_losses) > 1 else speaker_mel_losses[0]
            reg_loss = lambda_reg * torch.sum(delta_masked ** 2)

            total_loss = mel_loss + reg_loss

            if step == 0:
                initial_loss = total_loss.item()
            final_loss = total_loss.item()

            if not torch.isfinite(total_loss):
                logger.error(f"Step {step} total_loss is non-finite: mel_loss={mel_loss.item()}, reg_loss={reg_loss.item()}")
                break

            total_loss.backward()

            # Clean gradients and apply L_2 gradient clipping ||∇_δ||_2 ≤ max_grad_norm (Paper Section 3.2)
            if delta.grad is not None:
                torch.nan_to_num_(delta.grad, nan=0.0, posinf=max_grad_norm, neginf=-max_grad_norm)
                grad_norm = delta.grad.norm().item()
                grad_history.append(grad_norm)
                torch.nn.utils.clip_grad_norm_([delta], max_norm=max_grad_norm)
            else:
                grad_history.append(0.0)

            optimizer.step()

            # Enforce masking & norm projection constraint
            with torch.no_grad():
                delta.mul_(mask)
                cur_delta_norm = torch.norm(delta[:, target_indices, :]).item()
                if cur_delta_norm > max_delta_norm:
                    scale = max_delta_norm / cur_delta_norm
                    delta.mul_(scale)

            if step % 10 == 0 or step == n_steps - 1:
                cur_norm = torch.norm(delta[:, target_indices, :]).item()
                rel_ratio = cur_norm / base_target_norm
                logger.info(
                    f"[Step {step:02d}/{n_steps}] Total Loss: {total_loss.item():.4f} | "
                    f"Mel Loss: {mel_loss.item():.4f} | ||δ||: {cur_norm:.4f} | "
                    f"∇δ: {grad_norm:.4f} | Rel Ratio: {rel_ratio:.4f} | LR: {current_lr:.6f}"
                )

        # Final candidate perturbation δ*
        with torch.no_grad():
            final_delta = (delta * mask).detach()
            final_norm = torch.norm(final_delta[:, target_indices, :]).item()
            final_rel_ratio = final_norm / base_target_norm

            # Compute pooled key K_i = pool(c_I) and value V_i = pool(δ*_I) (Paper Eq. 5)
            if len(target_indices) > 0:
                key_pooled = base_embeddings[0, target_indices, :].mean(dim=0)     # [d]
                value_pooled = final_delta[0, target_indices, :].mean(dim=0)      # [d]
                idx_min = max(0, min(target_indices))
                idx_max = min(final_delta.shape[1], max(target_indices) + 1)
                word_delta = final_delta[:, idx_min:idx_max, :].clone()
            else:
                key_pooled = base_embeddings[0].mean(dim=0)                       # [d]
                value_pooled = final_delta[0].mean(dim=0)                         # [d]
                word_delta = final_delta.clone()

        converged = final_loss < initial_loss
        logger.info(
            f"✓ FlowEdit Latent Optimization Complete: Initial Loss={initial_loss:.4f} → "
            f"Final Loss={final_loss:.4f}, δ* Norm={final_norm:.4f}, Rel Ratio={final_rel_ratio:.4f}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return OptimizationResult(
            delta=final_delta,
            delta_pooled=value_pooled,
            key_pooled=key_pooled,
            target_token_indices=target_indices,
            initial_loss=initial_loss,
            final_loss=final_loss,
            total_steps=n_steps,
            delta_norm=final_norm,
            relative_delta_ratio=final_rel_ratio,
            converged=converged,
            grad_history=grad_history,
            base_embeddings=base_embeddings,
            word_delta=word_delta,
        )
