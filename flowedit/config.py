"""
FlowEdit Configuration — Research Paper Exact Hyperparameters.

Reference: FlowEdit (arXiv:2606.20518), Sections 3.1, 3.2, 4.1 & 4.5.
Flow Matching Text-to-Speech (F5-TTS DiT) Backbone.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Any
import os
import torch

from flowedit.utils.env import load_flowedit_env
load_flowedit_env()


@dataclass
class OptimizationConfig:
    """Stage 2: Latent input optimization parameters (Paper Section 3.2 & 4.1).

    δ* = argmin_δ [ ||Mel(g_θ(c + δ)) - Mel(y_ref)||_2^2 + λ||δ||_2^2 ]
    """

    # Number of Adam optimization steps (Paper Section 3.2: 50-100 steps)
    n_steps: int = 100

    # Learning rate schedule: cosine anneal from η0 = 0.080 → η_end = 0.002
    lr_start: float = 0.080
    lr_end: float = 0.002

    # L2 regularization weight on δ (low to allow full phonetic override from reference audio)
    lambda_reg: float = 0.0001

    # Gradient clipping L_2 max norm (Paper Section 3.2: ||∇_δ||_2 ≤ 1.0)
    grad_clip_max_norm: float = 1.0

    # Number of Euler ODE solver steps (Paper Section 3.1 & 3.2: N = 16 steps)
    ode_steps: int = 16

    # Relative perturbation norm constraint: ||δ_I|| / ||c_I|| ≤ max_relative_delta (1.2 allows full phonetic override)
    max_relative_delta: float = 1.20


    # Data augmentation on reference mel during optimization (Paper Section 3.2)
    enable_augmentation: bool = False
    augment_time_stretch_range: Tuple[float, float] = (0.9, 1.1)
    augment_gain_db_range: Tuple[float, float] = (-3.0, 3.0)

    # Optional F0 pitch guidance loss for tonal languages (Paper Section 4.5: α = 0.3)
    use_f0_loss: bool = False
    f0_loss_alpha: float = 0.3

    # Composite Spectral Loss Weights for crystal-clear phoneme pronunciation
    mel_l1_loss_weight: float = 1.0
    mel_mse_loss_weight: float = 0.5
    spectral_convergence_loss_weight: float = 0.2
    smooth_boundary_taper: bool = True


@dataclass
class MemoryConfig:
    """Stage 3: Modern Hopfield Network memory parameters (Paper Section 3.2 & Eq. 5, 6, 7).

    K_i = pool(c_I), V_i = pool(δ*_I)
    Mem(Q) = softmax(β Q K^T) V, β = 1/√d
    c_hat = c + σ(max_j(β Q K_j^T) - τ) ⊙ Mem(Q)
    """

    # Maximum number of stored corrections (Paper Section 3.2 & 4.5: M_max = 500)
    max_entries: int = 500

    # Deduplication & cluster averaging: cosine similarity >= 0.65 triggers running average for same contextual sense
    dedup_cosine_threshold: float = 0.65

    # Contextual homograph threshold: similarity below this for the same word creates a distinct contextual entry
    homograph_sim_threshold: float = 0.65

    # EMA decay α for deduplication merges (α = 0.80 for smooth exponential moving average)
    dedup_ema_decay: float = 0.80

    # EMA warm-up: while a cluster has at most this many members, the newest correction dominates
    # (decay = ema_warmup_decay) so early re-corrections replace a poor first reference quickly.
    ema_warmup_count: int = 2
    ema_warmup_decay: float = 0.20

    # Hopfield inverse temperature β = 1/√d (Paper Eq. 6: auto-computed from embedding_dim if None)
    hopfield_beta: Optional[float] = None

    # Score scale applied to cosine similarities when β < 1 (keeps sigmoid gating in a usable range)
    retrieval_score_scale: float = 10.0

    # Minimum per-position gate for the text-free continuous retrieval path (Paper Eq. 7)
    gate_activation_floor: float = 0.50

    # Learned gate threshold scalar τ (Calibrated to 4.0 for robust target word matching across sentences)
    gate_threshold_init: float = 4.0

    # ─── Contextual sense prototypes (EMA homograph disambiguation) ───
    # Every memory entry is a sense cluster whose prototype is an EMA of contextual occurrence
    # embeddings. The encoder is backbone-independent, so XTTS and F5-TTS share the same senses.
    # Empty string disables the encoder; disambiguation then falls back to backbone context keys.
    sense_encoder_model: str = field(
        default_factory=lambda: os.environ.get("FLOWEDIT_SENSE_ENCODER", "sentence-transformers/all-MiniLM-L6-v2")
    )
    # Hidden-state layers averaged for the occurrence embedding (middle layers carry word sense best)
    sense_encoder_layers: Tuple[int, ...] = (3, 4, 5)
    sense_encoder_device: str = field(default_factory=lambda: os.environ.get("FLOWEDIT_SENSE_DEVICE", "cpu"))
    # Words of context kept on each side of an occurrence, bounded by clause punctuation, so a
    # second occurrence in the same sentence is not contaminated by the first one's context
    sense_context_words: int = 6
    sense_clause_delimiters: str = ",;:.!?"
    sense_cache_size: int = 256
    # Same-pronunciation-unknown corrections merge into a cluster at or above this prototype similarity
    sense_merge_threshold: float = 0.90
    # A word with a single learned sense applies it only when the occurrence is at least this similar
    # (calibrated on held-out contexts: 30/31 same-sense applied, 15/21 other-sense rejected)
    sense_match_threshold: float = 0.84
    # A word with several learned senses picks the most similar one, provided it reaches this floor
    sense_min_similarity: float = 0.60
    # Encoder-unavailable fallback: minimum distance-weighted word-overlap for a single registered sense
    sense_lexical_threshold: float = 0.25

    # Context window in words for homograph disambiguation (Paper Section 3.2: ±1-3 words)
    context_window: int = 3

    # Context character radius for character-level tokenizers (encompassing surrounding words)
    context_char_radius: int = 4

    # Gaussian standard deviation for context key weighting across neighbouring tokens
    context_sigma: float = 2.0

    # Inference amplification factor on retrieved perturbation δ (Paper Eq. 7: calibrated 1.0 for continuous natural prosody)
    correction_scale: float = 1.0

    # Contextual averaging weights for memory key computation (must sum to 1.0)
    # K_i = α · word_emb + β · local_context + γ · global_context
    # Word embedding weight α: primary signal anchoring the key to the target word
    context_key_word_weight: float = 0.70
    # Local context weight β: Gaussian-weighted neighbors for sense disambiguation (e.g., "bank" in "river bank" vs "bank account")
    context_key_local_weight: float = 0.20
    # Global context weight γ: coarse sentence-level signal for broad disambiguation
    context_key_global_weight: float = 0.10

    # LRU pruning access age threshold
    lru_max_age: int = 1000



@dataclass
class AlignmentConfig:
    """Stage 1: Whisper forced alignment parameters (Paper Section 3.2)."""

    # Whisper model for alignment (Paper Section 3.2: Whisper-Large-v3, fallback to base if large unavailable)
    whisper_model: str = "base"

    # Token boundary expansion: 0 for exact target word tokens (avoids bleeding into neighbor spaces/words)
    token_expand: int = 0


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
    """TTS Backbone configuration supporting XTTS-v2 and F5-TTS."""

    backbone_type: str = "xtts"

    # XTTS Model checkpoint & configuration paths
    xtts_model_dir: str = field(
        default_factory=lambda: os.getenv(
            "FLOWEDIT_MODEL_DIR",
            os.getenv(
                "FLOWEDIT_XTTS_DIR",
                "/home/rsurya/projects/flow_edit/Flowedit/model"
                if os.path.isdir("/home/rsurya/projects/flow_edit/Flowedit/model")
                else ""
            )
        )
    )
    xtts_checkpoint: str = "model.pth"
    xtts_config_file: str = "config.json"
    xtts_vocab_file: str = "vocab.json"
    xtts_dvae_file: str = "dvae.pth"

    # F5-TTS Model checkpoint & vocab paths (optional overrides)
    f5tts_ckpt_file: str = ""
    f5tts_vocab_file: str = ""
    vocoder_local_path: str = ""
    f5tts_model_name: str = ""  # e.g. "F5TTS_Base" or "F5TTS_v1_Base"

    # Acoustic safety bounds on injected text-embedding perturbations: element-wise clamp and
    # L2 cap over the active positions (prevents out-of-distribution noise)
    xtts_delta_clamp: float = 0.30
    xtts_delta_norm_cap: float = 4.0
    f5tts_delta_clamp: float = 0.35
    f5tts_delta_norm_cap: float = 3.5

    # Device & dtype
    device: str = field(default_factory=lambda: os.environ.get("FLOWEDIT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    optimization_dtype: str = "float32"
    inference_dtype: str = "float32"


@dataclass
class S3Config:
    """S3 Bucket configuration for deterministic phonetic spelling corrections."""

    bucket_name: str = field(default_factory=lambda: os.environ.get("FLOWEDIT_S3_BUCKET", os.environ.get("S3_BUCKET_NAME", os.environ.get("AWS_S3_BUCKET", "flowedit-bucket"))))
    endpoint_url: Optional[str] = field(default_factory=lambda: os.environ.get("FLOWEDIT_S3_URL", os.environ.get("S3_ENDPOINT_URL", None)))
    prefix: str = field(default_factory=lambda: os.environ.get("FLOWEDIT_S3_PREFIX", "corrections/"))
    region_name: str = field(default_factory=lambda: os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")))


@dataclass
class FlowEditConfig:
    """Master configuration combining all FlowEdit modules."""

    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    s3: S3Config = field(default_factory=S3Config)

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
