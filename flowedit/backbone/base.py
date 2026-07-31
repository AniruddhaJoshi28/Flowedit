"""
Abstract TTS Backbone Interface for FlowEdit.

This module defines the unified TTSBackbone interface. All backbones (F5-TTS, XTTS-v2, etc.)
implement this abstract base class.

The FlowEdit pipeline interacts ONLY through this interface, ensuring that the paper's
core algorithm (Whisper alignment -> Latent Optimization -> Hopfield Memory -> Gated Retrieval)
remains 100% identical and model-agnostic.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any


class TTSBackbone(nn.Module, ABC):
    """Abstract Base Class for FlowEdit TTS Backbones."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimension d of the text embedding space."""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Return the device (cuda/cpu) where the model is loaded."""
        pass

    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        """Return the tokenizer instance for token-text mapping."""
        pass

    @property
    @abstractmethod
    def optimization_mode(self) -> str:
        """Return the optimization mode used by this backbone for delta learning.
        
        Supported modes:
            - 'adjoint_ode': Continuous flow-matching adjoint ODE solver (F5-TTS)
            - 'teacher_forcing': Teacher-forced autoregressive cross-entropy loss (XTTS-v2)
        """
        pass

    @property
    def supports_differentiable_optimization(self) -> bool:
        """Whether this backbone supports direct end-to-end autograd/adjoint gradients."""
        return True

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights and initialize internal components."""
        pass

    @abstractmethod
    def get_token_ids(self, text: str, language: str = "en") -> torch.Tensor:
        """Tokenize text string into token IDs [1, S]."""
        pass

    @abstractmethod
    def encode_text(self, text: str, language: str = "en") -> torch.Tensor:
        """Encode text string into text embeddings c ∈ R^[1, S, d].
        
        Used by Hopfield Memory for key computation and as base for perturbation δ.
        """
        pass

    @abstractmethod
    def get_speaker_embedding(
        self,
        audio_path: Optional[str] = None,
        language: str = "en",
        ref_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract speaker conditioning from reference audio file."""
        pass

    @abstractmethod
    def compute_optimization_loss(
        self,
        perturbed_embeddings: torch.Tensor,
        ref_audio_path: str,
        speaker_conditioning: Dict[str, Any],
        text: str,
        language: str = "en",
        target_word_start_time: Optional[float] = None,
        target_word_end_time: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute backbone-specific loss for optimizing perturbation δ.
        
        - F5-TTS: Mel-spectrogram L2 reconstruction loss (Eq. 3 in paper)
        - XTTS: Cross-entropy loss on next-token logits via teacher forcing
        
        The optimizer calls ONLY this method without any model-specific branches.
        """
        pass

    @abstractmethod
    def synthesize_from_embeddings(
        self,
        text_embeddings: torch.Tensor,
        speaker_conditioning: Dict[str, Any],
        text: str,
        language: str = "en",
    ) -> Tuple[torch.Tensor, int]:
        """Synthesize audio using (possibly perturbed) text embeddings c + δ.
        
        Returns:
            Tuple of (waveform tensor [1, T], sample_rate int)
        """
        pass

    @abstractmethod
    def synthesize_direct(
        self,
        text: str,
        speaker_conditioning: Dict[str, Any],
        language: str = "en",
        user_ref_text: Optional[str] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Direct synthesis without embedding hooks (when Hopfield gate is inactive).
        
        Returns:
            Tuple of (waveform tensor [1, T], sample_rate int)
        """
        pass
