"""
FlowEdit Configuration — all hyperparameters from the paper.

Reference: FlowEdit (arXiv:2606.20518), Section 3.2 & 4.1
Adapted for XTTS-2 backbone (autoregressive GPT, not flow-matching).
"""

from dataclasses import dataclass, field
from typing import Optional
import os
import torch


@dataclass
class OptimizationConfig:
    """Stage 2: Latent input optimization parameters.

    These control how the perturbation δ is optimized to match the
    reference pronunciation. Values from paper Section 3.2.
    """

    # Number of Adam optimization steps (paper: 50, tuned: 80 for full convergence)
    n_steps: int = 80

    # Learning rate schedule: cosine anneal from lr_start → lr_end
    lr_start: float = 0.005
    lr_end: float = 0.0005

    # L2 regularization weight on δ (paper: λ=0.001)
    lambda_reg: float = 0.01

    # Gradient clipping max norm (paper: ‖∇_δ‖_∞ ≤ 1.0)
    grad_clip_max_norm: float = 1.0

    # Data augmentation on reference mel during optimization
    augment_time_stretch_range: tuple = (0.9, 1.1)
    # F0 pitch guidance loss (paper Section 4.5: optional extension)
    f0_loss_alpha: float = 0.3
    use_f0_loss: bool = False

    augment_gain_db_range: tuple = (-3.0, 3.0)
    enable_augmentation: bool = True



@dataclass
class MemoryConfig:
    """Stage 3: Modern Hopfield Network memory parameters.

    Controls the associative memory that stores pronunciation corrections.
    Reference: paper Section 3.2 (Stage 3) and Section 4.5.
    """

    # Maximum number of stored corrections
    max_entries: int = 500

    # Deduplication: cosine similarity > threshold → EMA update
    dedup_cosine_threshold: float = 0.95

    # EMA decay for deduplication merges
    dedup_ema_decay: float = 0.9

    # Hopfield inverse temperature β = 1/√d (auto-computed if None)
    hopfield_beta: Optional[float] = None

    # Learned gate threshold τ initialization
    gate_threshold_init: float = 0.4

    # Perturbation scale factor to amplify learned phonetic corrections
    # (1.0 for XTTS autoregressive GPT to avoid phonetic distortion / stuttering)
    perturbation_scale: float = 1.0

    # Context window for homograph disambiguation (paper: ±3 tokens)
    # Keys are Gaussian-weighted average of surrounding embeddings
    context_window: int = 3

    # Gaussian std for context weighting
    context_sigma: float = 1.5

    # LRU pruning: entries not accessed in this many retrievals are pruned
    lru_max_age: int = 1000


@dataclass
class AlignmentConfig:
    """Stage 1: Whisper forced alignment parameters.

    Controls how reference audio is aligned to extract target token indices.
    """

    # Whisper model size for alignment
    whisper_model: str = "base"

    # Token boundary expansion: ±N tokens around detected target
    # Absorbs tokenizer boundary errors (paper: ±1)
    token_expand: int = 1

    # Minimum confidence for alignment acceptance
    min_confidence: float = 0.5

    # Language hint for Whisper (None = auto-detect)
    language: Optional[str] = None


@dataclass
class AudioConfig:
    """Audio processing parameters."""

    # Sample rate for all audio processing
    # F5-TTS operates natively at 24000 Hz — MUST match backbone
    sample_rate: int = 24000

    # Mel-spectrogram parameters
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    fmin: float = 0.0
    fmax: Optional[float] = 8000.0

    # Reference audio constraints (paper: ≥1.5s optimal, plateaus >3s)
    # Lowered min to 0.3s because reference audio of a single word can be short
    ref_audio_min_duration: float = 0.3
    ref_audio_max_duration: float = 10.0


@dataclass
class BackboneConfig:
    """Backbone configuration supporting XTTS-v2, F5-TTS, and CosyVoice (2/3).

    Set backbone_type to select which model to use:
        - "xtts"      : Use local XTTS-v2 model from xtts_model_dir
        - "f5tts"     : Use F5-TTS via f5-tts package
        - "cosyvoice" : Use CosyVoice / CosyVoice 2 / CosyVoice 3
    """

    # Backbone selector: "xtts", "f5tts", or "cosyvoice"
    backbone_type: str = "cosyvoice"

    # ── XTTS-specific settings ──────────────────────────────────────
    xtts_model_dir: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "Xtts"
    )
    xtts_checkpoint: str = "model.pth"

    # ── F5-TTS-specific settings ────────────────────────────────────
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"

    # ── CosyVoice-specific settings ──────────────────────────────────
    cosyvoice_model_version: str = "2"   # "2", "3", "300M", "0.5B"
    cosyvoice_model_dir: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "CosyVoice"
    )
    cosyvoice_mode: str = "zero_shot"    # "zero_shot", "cross_lingual", "instruct"

    # ── Shared settings ─────────────────────────────────────────────
    use_gradient_checkpointing: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    optimization_dtype: str = "float32"
    inference_dtype: str = "float16"



@dataclass
class FlowEditConfig:
    """Master configuration combining all sub-configs.

    Usage:
        config = FlowEditConfig()
        config.optimization.n_steps = 100  # Override defaults
    """

    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)

    # Global settings
    seed: int = 42
    verbose: bool = True

    def __post_init__(self):
        """Auto-compute derived values."""
        # Auto-set Hopfield β if not specified
        # Paper: β = 1/√d where d is embedding dimension
        # XTTS-2 embedding dim is set after model loading
        pass

    @classmethod
    def from_yaml(cls, path: str) -> "FlowEditConfig":
        """Load configuration from a YAML file."""
        import yaml
        with open(path, "r") as f:
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
        """Save configuration to a YAML file."""
        import yaml
        from dataclasses import asdict

        data = asdict(self)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
