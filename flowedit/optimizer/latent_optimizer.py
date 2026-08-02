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
        ref_audio_path: str,
        token_indices: List[int],
        speaker_conditioning: dict,
        language: str = "en",
        target_word_start_time: Optional[float] = None,
        target_word_end_time: Optional[float] = None,
        progress_callback: Optional[Callable] = None,
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
            target_word_start_time: Start time of target word in seconds
                                   (from Whisper alignment of reference audio)
            target_word_end_time: End time of target word in seconds
            progress_callback: Optional callback(step, loss, grad_norm)

        Returns:
            OptimizationResult with the optimized δ*
        """
        device = backbone.device

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

        # Cosine annealing: η₀=0.01 → η₅₀=0.001
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

        progress = tqdm(
            range(self.config.n_steps),
            desc="Optimizing δ",
            disable=not logger.isEnabledFor(logging.INFO),
        )

        for step in progress:
            optimizer.zero_grad()

            # Apply mask: zero out non-target positions
            # This is the key constraint from the paper
            delta_masked = delta * mask

            # Perturbed embeddings: c + δ
            perturbed_embeddings = base_embeddings.detach() + delta_masked

            # Compute backbone-specific optimization loss (differentiable path)
            try:
                loss_dict = backbone.compute_optimization_loss(
                    perturbed_embeddings=perturbed_embeddings,
                    ref_audio_path=ref_audio_path,
                    speaker_conditioning=speaker_conditioning,
                    text=text,
                    language=language,
                    target_word_start_time=target_word_start_time,
                    target_word_end_time=target_word_end_time,
                )
                task_loss = loss_dict["loss"]
            except Exception as e:
                logger.warning(f"Optimization loss computation failed at step {step}: {e}")
                task_loss = torch.tensor(0.0, device=device, requires_grad=True)
                loss_dict = {"loss": task_loss}

            # ── Diagnostic checks ──
            if step == 0:
                logger.info(
                    f"  [DIAG] task_loss.requires_grad={task_loss.requires_grad}, "
                    f"task_loss={task_loss.item():.6f}"
                )
                if not task_loss.requires_grad:
                    logger.error(
                        "  [DIAG] ⚠ task_loss has NO gradient! "
                        "The optimization will not produce meaningful δ values."
                    )
                if abs(task_loss.item()) < 1e-8:
                    logger.warning(
                        "  [DIAG] ⚠ task_loss is ~0.0 — "
                        "likely hitting the dummy fallback loss path."
                    )

            # Regularization: ||δ||² (prevent catastrophic forgetting / excessive deviation)
            reg_loss = self.config.lambda_reg * torch.sum(delta_masked ** 2)

            # Total loss: L = L_task + λ||δ||²
            total_loss = task_loss + reg_loss

            # Backward pass (autograd/adjoint through frozen backbone)
            total_loss.backward()

            # ── Gradient diagnostic (first step only) ──
            if step == 0 and delta.grad is not None:
                delta_grad_norm = delta.grad.norm().item()
                delta_grad_max = delta.grad.abs().max().item()
                logger.info(
                    f"  [DIAG] δ grad norm={delta_grad_norm:.6f}, "
                    f"δ grad max={delta_grad_max:.6f}"
                )
                if delta_grad_norm < 1e-8:
                    logger.error(
                        "  [DIAG] ⚠ δ gradient is near-zero! "
                        "No meaningful perturbation will be learned."
                    )

            # Gradient clipping: ||∇_δ||_∞ ≤ 1.0
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [delta], self.config.grad_clip_max_norm
            )

            # Step optimizer
            optimizer.step()
            scheduler.step()
            
            if step % 10 == 0 or step == self.config.n_steps - 1:
                delta_norm = torch.norm(delta_masked).item()
                diag_str = " | ".join(f"{k}: {v:.4f}" for k, v in loss_dict.items() if k != "loss" and isinstance(v, (int, float)))
                logger.info(
                    f"Step {step:02d} | "
                    f"Task Loss: {task_loss.item():.4f} | "
                    f"Reg Loss: {reg_loss.item():.4f} | "
                    f"Total Loss: {total_loss.item():.4f} | "
                    f"||δ||: {delta_norm:.4f} | "
                    f"||∇δ||: {grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                    f"{diag_str}"
                )

            # Re-apply mask after gradient step (ensure constraint)
            with torch.no_grad():
                delta.data *= mask

            # Track metrics
            loss_val = total_loss.item()
            grad_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            loss_history.append(loss_val)
            grad_norm_history.append(grad_val)

            # Progress bar update
            progress.set_postfix({
                "loss": f"{loss_val:.4f}",
                "task": f"{task_loss.item():.4f}",
                "reg": f"{reg_loss.item():.4f}",
                "∇": f"{grad_val:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.5f}",
            })

            if progress_callback:
                progress_callback(step, loss_val, grad_val, perturbed_embeddings)

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
