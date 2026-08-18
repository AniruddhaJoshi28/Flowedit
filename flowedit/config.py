"""
FlowEdit Configuration — Research Paper Exact Hyperparameters.

Reference: FlowEdit (arXiv:2606.20518), Sections 3.1, 3.2, 4.1 & 4.5.
Flow Matching Text-to-Speech (F5-TTS DiT) Backbone.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Any
import os
import torch


@dataclass
class OptimizationConfig:
    """Stage 2: Latent input optimization parameters (Paper Section 3.2 & 4.1).

    δ* = argmin_δ [ ||Mel(g_θ(c + δ)) - Mel(y_ref)||_2^2 + λ||δ||_2^2 ]
    """

    # Number of Adam optimization steps (Paper Section 3.2: 50-100 steps)
    n_steps: int = 50

    # Learning rate schedule: cosine anneal from η0 = 0.02 → η_end = 0.002
    lr_start: float = 0.02
    lr_end: float = 0.002

    # L2 regularization weight on δ (Paper Section 3.2 & Table 2: λ = 0.0005)
    lambda_reg: float = 0.0005

    # Gradient clipping L_infinity max norm (Paper Section 3.2: ||∇_δ||_∞ ≤ 1.0)
    grad_clip_max_norm: float = 1.0

    # Number of Euler ODE solver steps (Paper Section 3.1 & 3.2: N = 16 steps for fast/memory-efficient differentiation)
    ode_steps: int = 16

    # Relative perturbation norm constraint: ||δ_I|| / ||c_I|| ≤ max_relative_delta
    max_relative_delta: float = 3.0

    # Data augmentation on reference mel during optimization (Paper Section 3.2)
    enable_augmentation: bool = False
    augment_time_stretch_range: Tuple[float, float] = (0.9, 1.1)
    augment_gain_db_range: Tuple[float, float] = (-3.0, 3.0)

    # Optional F0 pitch guidance loss for tonal languages (Paper Section 4.5: α = 0.3)
    use_f0_loss: bool = False
    f0_loss_alpha: float = 0.3


@dataclass
class MemoryConfig:
    """Stage 3: Modern Hopfield Network memory parameters (Paper Section 3.2 & Eq. 5, 6, 7).

    K_i = pool(c_I), V_i = pool(δ*_I)
    Mem(Q) = softmax(β Q K^T) V, β = 1/√d
    c_hat = c + σ(max_j(β Q K_j^T) - τ) ⊙ Mem(Q)
    """

    # Maximum number of stored corrections (Paper Section 3.2 & 4.5: M_max = 500)
    max_entries: int = 500

    # Deduplication: cosine similarity > 0.95 triggers EMA update (Paper Section 3.2)
    dedup_cosine_threshold: float = 0.95

    # EMA decay α for deduplication merges (Paper Section 3.2: α = 0.90)
    dedup_ema_decay: float = 0.90

    # Hopfield inverse temperature β = 1/√d (Paper Eq. 6: auto-computed from embedding_dim if None)
    hopfield_beta: Optional[float] = None

    # Learned gate threshold scalar τ (Paper Section 3.2: τ ≈ 9.0 for precise homograph gating)
    gate_threshold_init: float = 9.0

    # Context window for homograph disambiguation (Paper Section 3.2: ±1-2 tokens)
    context_window: int = 1

    # Gaussian standard deviation for context key weighting (Paper Section 3.2)
    context_sigma: float = 0.8

    # Inference amplification factor on retrieved perturbation δ (Paper Eq. 7 gain)
    correction_scale: float = 1.5

    # LRU pruning access age threshold
    lru_max_age: int = 1000


@dataclass
class AlignmentConfig:
    """Stage 1: Whisper forced alignment parameters (Paper Section 3.2)."""

    # Whisper model for alignment (Paper Section 3.2: Whisper-Large-v3, fallback to base if large unavailable)
    whisper_model: str = "base"

    # Token boundary expansion: ±1 token around detected target (Paper Section 3.2)
    token_expand: int = 1

    # Minimum alignment confidence threshold
    min_confidence: float = 0.5

    # Language hint (None = auto-detect)
    language: Optional[str] = None


@dataclass
class AudioConfig:
    """Audio and Mel-spectrogram processing parameters."""

    # F5-TTS native sampling rate
    sample_rate: int = 24000

    # Mel-spectrogram parameters
    n_mels: int = 100
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    fmin: float = 0.0
    fmax: Optional[float] = None

    # Reference audio constraints (Paper Section 4.2: ≥1.5s optimal, plateaus >3s)
    ref_audio_min_duration: float = 0.2
    ref_audio_max_duration: float = 15.0


@dataclass
class BackboneConfig:
    """F5-TTS Diffusion Transformer Backbone configuration."""

    backbone_type: str = "f5tts"

    # Model checkpoint & vocab paths (optional overrides, downloads automatically if empty)
    f5tts_ckpt_file: str = ""
    f5tts_vocab_file: str = ""
    vocoder_local_path: str = ""

    # Device & dtype
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    optimization_dtype: str = "float32"
    inference_dtype: str = "float32"


@dataclass
class FlowEditConfig:
    """Master configuration combining all FlowEdit modules."""

    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)

    seed: int = 42
    verbose: bool = True

    def __post_init__(self):
        # Propagate device preference
        if hasattr(self.backbone, "device"):
            pass

    @classmethod
    def from_yaml(cls, path: str) -> "FlowEditConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        config = cls()
        for section_name, section_data in data.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                section = getattr(config, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
            elif hasattr(config, section_name):
                setattr(config, section_name, section_data)
        return config

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        from dataclasses import asdict

        data = asdict(self)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
