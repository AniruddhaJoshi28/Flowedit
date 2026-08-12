"""
Latent Input Optimizer — Stage 2 of FlowEdit (The Core Innovation).

Paper Section 3.2 (Stage 2: Latent Input Optimization):
    "We freeze all DiT parameters θ and introduce a learnable perturbation
    δ ∈ R^(S×d) initialized at zero for sequence length S:
        δ* = argmin_δ ||Mel(g_θ(c + δ)) - Mel(y_ref)||² + λ||δ||²
    where c = E(x) denotes text-encoder embeddings, g_θ denotes synthesis
    through the frozen DiT and ODE solver, and λ=0.001 is a regularization
    weight. We mask non-target positions (δ_j = 0 ∀ j ∉ I)."

XTTS-2 Adaptation:
    - F5-TTS uses adjoint sensitivity method through ODE (constant memory)
    - XTTS-2 uses standard autograd through GPT with gradient checkpointing
    - The optimization objective is identical; only the gradient path differs

Convergence Pattern (paper Section 4.3):
    - Rapid descent (steps 1-15): ||∇_δL|| drops 85%, coarse phonetics
    - Refinement (steps 15-50): gradient norms < 0.02, spectral details
    - Consistent across all 312 test words
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import logging
from dataclasses import dataclass
from typing import Optional, List, Callable
from tqdm import tqdm

from flowedit.config import OptimizationConfig, AudioConfig
from flowedit.utils.audio import AudioProcessor
from flowedit.utils.metrics import compute_mel_loss, compute_f0_loss

logger = logging.getLogger(__name__)


TRUST_REGION_RATIO = {
    "f5_conditioning": 0.08,
    "xtts_gpt_conditioning": 0.02,
    "cosyvoice_adapter_hidden": 0.01,
}


def project_delta(
    delta: torch.Tensor,
    base: torch.Tensor,
    ratio: float = 0.03,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Relative trust-region projection function.

    Clamps perturbation delta so its L2 norm does not exceed ratio * base_norm.
    """
    delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(eps)
    base_norm = base.norm(dim=-1, keepdim=True).clamp_min(eps)
    max_norm = ratio * base_norm
    scale = torch.minimum(
        torch.ones_like(delta_norm),
        max_norm / delta_norm,
    )
    return delta * scale


@dataclass
class OptimizationResult:

    """Result of latent input optimization.

    Attributes:
        delta: Optimized perturbation vector δ* [1, seq_len, dim]
        delta_target: Only the target-token perturbation (pooled for memory storage)
        target_indices: Token indices that were optimized
        final_loss: Final optimization loss value
        loss_history: Loss at each optimization step
        grad_norm_history: Gradient norm at each step
        converged: Whether the optimization converged
    """
    delta: torch.Tensor
    delta_target: torch.Tensor
    target_indices: List[int]
    final_loss: float
    loss_history: List[float]
    grad_norm_history: List[float]
    converged: bool


class LatentOptimizer:
    """Optimizes perturbation δ in text embedding space.

    This is the core innovation of FlowEdit — instead of modifying the
    model weights (which causes catastrophic forgetting), we optimize a
    small perturbation to the text embeddings that corrects pronunciation.

    The optimization finds δ* such that synthesizing with (c + δ*) produces
    audio matching the reference pronunciation, while the L2 regularization
    ensures δ* stays close to zero (minimal perturbation principle).
    """

    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        audio_config: Optional[AudioConfig] = None,
    ):
        self.config = config or OptimizationConfig()
        self.audio_processor = AudioProcessor(audio_config)

    def optimize(
        self,
        backbone,
        text: str,
        target_word: str,
        ref_audio_path: str,
        token_indices: List[int],
        speaker_conditioning: dict,
        language: str = "en",
        target_word_start_sample: Optional[int] = None,
        target_word_end_sample: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ) -> OptimizationResult:
        """Run latent input optimization for pronunciation correction.

        Implements Eq. 3 from the paper:
            δ* = argmin_δ ||Mel(g_θ(c + δ)) - Mel(y_ref)||² + λ||δ||²

        CRITICAL: The mel loss must be computed between:
            - The TARGET WORD SEGMENT extracted from the synthesized audio
            - The reference audio (which contains only the target word)
        NOT between the full synthesized audio and the reference.

        Args:
            backbone: Frozen F5TTSBackbone instance
            text: Input text containing the target word
            ref_audio_path: Path to reference audio with correct pronunciation
            token_indices: Token indices to optimize (from Stage 1)
            speaker_conditioning: Speaker embedding dict from backbone
            language: Language code
            target_word_start_sample: Start sample index of target word in synthesized audio
            target_word_end_sample: End sample index of target word in synthesized audio
            progress_callback: Optional callback(step, loss, grad_norm)

        Returns:
            OptimizationResult with the optimized δ*
        """
        device = backbone.device

        # Backwards compatibility fallback for older keyword argument names
        if target_word_start_sample is None:
            target_word_start_sample = kwargs.get("target_word_start_idx")
        if target_word_end_sample is None:
            target_word_end_sample = kwargs.get("target_word_end_idx")

        # Step 1: Get base text embeddings c = E(text)
        logger.info("Step 1: Encoding base text embeddings")
        with torch.no_grad():
            base_embeddings = backbone.encode_text(text, language)  # [1, S, d]

        seq_len = base_embeddings.shape[1]
        embed_dim = base_embeddings.shape[2]

        logger.info(
            f"  Embedding shape: [{seq_len} tokens × {embed_dim} dim]"
        )
        logger.info(f"  Target token indices: {token_indices}")

        # Assertion B Check: Strict invariant validation of target token indices against F5 text tokenization
        try:
            full_token_ids = backbone.get_token_ids(text, language)[0].tolist()
            invalid_indices = [i for i in token_indices if i < 0 or i >= len(full_token_ids)]
            if invalid_indices:
                raise RuntimeError(
                    f"[Assertion B] Invalid F5 token indices: {invalid_indices}. "
                    f"F5 token sequence length={len(full_token_ids)}, token_indices={token_indices}"
                )
            target_token_ids = [full_token_ids[i] for i in token_indices]
            logger.info(f"  [Assertion B] F5 token sequence length={len(full_token_ids)}")
            logger.info(f"  [Assertion B] target positions={token_indices}")
            logger.info(f"  [Assertion B] target token IDs={target_token_ids}")
        except Exception as ex:
            if "Invalid F5 token indices" in str(ex):
                raise
            logger.warning(f"Could not log Assertion B token IDs: {ex}")

        logger.info("Step 2: Preparing reference mel-spectrogram")
        ref_waveform, sr = self.audio_processor.load_audio(ref_audio_path)
        ref_waveform = ref_waveform.to(device)
        ref_mel = self.audio_processor.compute_mel(ref_waveform)

        # Step 3: Initialize δ = zeros(S, d) — paper specification
        logger.info("Step 3: Initializing perturbation δ = 0")
        delta = torch.zeros(
            1, seq_len, embed_dim,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )

        # Create mask: only target tokens are optimizable
        # Paper: δ_j = 0 ∀ j ∉ I
        mask = torch.zeros(1, seq_len, 1, device=device)
        for idx in token_indices:
            if 0 <= idx < seq_len:
                mask[0, idx, 0] = 1.0

        # Step 4: Setup optimizer — Adam with cosine LR schedule
        optimizer = optim.Adam(
            [delta],
            lr=self.config.lr_start,
            betas=(0.9, 0.999),
        )

        # Cosine annealing schedule
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.n_steps,
            eta_min=self.config.lr_end,
        )

        # Step 5: Optimization loop
        logger.info(
            f"Step 5: Running {self.config.n_steps} optimization steps "
            f"(λ={self.config.lambda_reg}, lr={self.config.lr_start}→{self.config.lr_end})"
        )

        loss_history = []
        grad_norm_history = []
        task_loss_history = []
        
        initial_task_loss = None

        progress = tqdm(
            range(self.config.n_steps),
            desc="Optimizing δ",
            disable=not logger.isEnabledFor(logging.INFO),
        )

        for step in progress:
            optimizer.zero_grad()
            
            delta_before_step = delta.detach().clone()

            # Apply mask: zero out non-target positions
            delta_masked = delta * mask

            # Apply delta to the target tokens in the full sequence
            perturbed_embeddings = base_embeddings.detach() + delta_masked

            # Compute backbone-specific optimization loss
            try:
                loss_dict = backbone.compute_optimization_loss(
                    perturbed_embeddings=perturbed_embeddings,
                    ref_audio_path=ref_audio_path,
                    speaker_conditioning=speaker_conditioning,
                    text=text,
                    language=language,
                    target_word_start_sample=target_word_start_sample,
                    target_word_end_sample=target_word_end_sample,
                )
                task_loss = loss_dict["loss"]
            except Exception as e:
                import traceback
                logger.error(f"Optimization loss computation failed at step {step}: {e}")
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Optimization failed during F5-TTS synthesis: {e}") from e

            # Record initial loss for relative reduction check
            if step == 0:
                initial_task_loss = task_loss.item()
                if not task_loss.requires_grad:
                    raise RuntimeError("task_loss has NO gradient. Adjoint ODE solver failed to preserve gradients.")

            reg_loss = self.config.lambda_reg * torch.sum(delta_masked ** 2)
            total_loss = task_loss + reg_loss

            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Optimization loss became non-finite at step {step}.")

            total_loss.backward()

            # Diagnostic check: unmasked gradient flow across all token positions
            with torch.no_grad():
                unmasked_grad = delta.grad.abs().sum(dim=-1)[0] # [seq_len]
                target_grads = unmasked_grad[token_indices] if len(token_indices) > 0 else torch.tensor([])
                logger.info(f"[DIAG Step {step}] Unmasked max token grad: {unmasked_grad.max().item():.8f}, Target token grads: {target_grads.tolist()}")

            # Assert gradients are flowing properly
            assert delta.requires_grad, "delta must require grad"
            assert delta.grad is not None, "delta.grad is None after backward"
            if not torch.isfinite(delta.grad).all():
                raise FloatingPointError(f"delta.grad contains non-finite values at step {step}")

            grad_norm_val = delta.grad.norm().item()
            if step == 0 and grad_norm_val < 1e-10:
                logger.warning(f"Initial gradient norm is extremely small: {grad_norm_val}")

            # Paper spec: infinity norm clipping (||∇_δ||_∞ <= max_norm)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [delta], self.config.grad_clip_max_norm, norm_type=float("inf")
            )

            optimizer.step()
            scheduler.step()

            # Enforce target token masking after optimizer step (L2 regularization constrains delta norm)
            with torch.no_grad():
                delta.data *= mask
                if not torch.isfinite(delta).all():
                    raise FloatingPointError(f"Perturbation delta became non-finite at step {step}.")
            
            update_norm = (delta.detach() - delta_before_step).norm().item()
            delta_norm_val = delta.norm().item()
            with torch.no_grad():
                embedding_diff = (delta * mask).norm().item()

            if step % 10 == 0 or step == self.config.n_steps - 1:
                logger.info(
                    f"Step {step:02d} | "
                    f"task_loss={task_loss.item():.4f} | "
                    f"reg_loss={reg_loss.item():.4f} | "
                    f"total_loss={total_loss.item():.4f} | "
                    f"grad_norm={grad_norm_val:.6f} | "
                    f"delta_norm={delta_norm_val:.6f} | "
                    f"update_norm={update_norm:.6f} | "
                    f"embedding_diff={embedding_diff:.6f}"
                )

            # Track metrics
            loss_val = total_loss.item()
            task_loss_history.append(task_loss.item())
            loss_history.append(loss_val)
            grad_norm_history.append(grad_norm_val)

            progress.set_postfix({
                "task": f"{task_loss.item():.4f}",
                "∇": f"{grad_norm_val:.4f}",
                "Δ_norm": f"{delta_norm_val:.4f}",
            })

            if progress_callback:
                progress_callback(step, loss_val, grad_norm_val, perturbed_embeddings)

        # Post-optimization verification
        final_task_loss = task_loss_history[-1]
        min_task_loss = min(task_loss_history)
        relative_reduction = (initial_task_loss - final_task_loss) / max(abs(initial_task_loss), 1e-8)
        
        logger.info(
            f"Optimization finished. Initial Task Loss: {initial_task_loss:.6f}, "
            f"Final Task Loss: {final_task_loss:.6f}, "
            f"Min Task Loss: {min_task_loss:.6f}, "
            f"Relative Reduction: {relative_reduction:.2%}"
        )
        
        if relative_reduction <= 0 and min_task_loss >= initial_task_loss:
            logger.warning(
                f"Optimization failed to reduce task loss: "
                f"initial={initial_task_loss:.6f}, final={final_task_loss:.6f}"
            )

        # Step 6: Extract optimized δ* sequence
        with torch.no_grad():
            delta_final = (delta * mask).detach()

            # Preserve full sequence perturbation [n_target, d] instead of pooling
            # This is critical for character-level models like F5-TTS
            target_delta_vectors = delta_final[0, token_indices, :]  # [n_target, d]
            delta_target_sequence = target_delta_vectors

        # Check convergence (gradient norms < 0.02 per paper)
        converged = (
            len(grad_norm_history) > 15 and
            sum(grad_norm_history[-10:]) / 10 < 0.02
        )

        logger.info(
            f"Optimization complete. Final loss: {loss_history[-1]:.4f}, "
            f"Converged: {converged}, "
            f"δ norm: {torch.norm(delta_final).item():.4f}"
        )

        return OptimizationResult(
            delta=delta_final,
            delta_target=delta_target_sequence,
            target_indices=token_indices,
            final_loss=loss_history[-1],
            loss_history=loss_history,
            grad_norm_history=grad_norm_history,
            converged=converged,
        )
