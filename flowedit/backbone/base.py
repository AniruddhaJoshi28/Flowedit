"""
Abstract TTS Backbone Interface for FlowEdit.

This module defines the unified TTSBackbone interface. All backbones (F5-TTS, XTTS-v2, etc.)
implement this abstract base class.

The FlowEdit pipeline interacts ONLY through this interface, ensuring that the paper's
core algorithm (Whisper alignment -> Latent Optimization -> Hopfield Memory -> Gated Retrieval)
remains 100% identical and model-agnostic.
"""

from abc import ABC, abstractmethod
from enum import Enum
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any



from dataclasses import dataclass

class OptimizationMode(str, Enum):
    """Supported backbone optimization modes."""
    FLOW_MATCHING = "flow_matching"
    AUTOREGRESSIVE = "autoregressive"
    DIFFUSION = "diffusion"
    OTHER = "other"


@dataclass
class BackboneCapabilities:
    """Explicit capabilities descriptor for a TTS backbone."""
    differentiable_embeddings: bool = True
    differentiable_decoder: bool = True
    teacher_forcing: bool = False
    supports_embedding_hook: bool = True
    supports_gradient_checkpointing: bool = True


class TTSBackbone(nn.Module, ABC):
    """Abstract Base Class for FlowEdit TTS Backbones."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config

    @property
    def capabilities(self) -> BackboneCapabilities:
        """Return explicit BackboneCapabilities descriptor."""
        return BackboneCapabilities()


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
    def optimization_mode(self) -> OptimizationMode:
        """Return the OptimizationMode enum for this backbone."""
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
    def tokenize(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Tokenize text string into token dictionary containing 'token_ids' and metadata."""
        pass

    @abstractmethod
    def detokenize(self, token_ids: torch.Tensor) -> str:
        """Convert token IDs back into text representation."""
        pass

    def get_token_ids(self, text: str, language: str = "en") -> torch.Tensor:
        """Backward-compatible helper: Tokenize text string into token IDs [1, S]."""
        res = self.tokenize(text, language=language)
        if isinstance(res, dict) and "token_ids" in res:
            return res["token_ids"]
        if isinstance(res, torch.Tensor):
            return res
        return torch.tensor(res, dtype=torch.long)

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
    ) -> Dict[str, Any]:
        """Compute backbone-specific differentiable loss for optimizing perturbation δ.
        
        Returns:
            Dict containing at least:
                - "loss": A scalar torch.Tensor containing the total loss to backpropagate.
                - "ce_loss" or "mel_loss": The primary loss term.
                - "delta_norm": Norm of the perturbation.
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

    def decode_embeddings(
        self,
        text_embeddings: torch.Tensor,
        speaker_conditioning: Dict[str, Any],
        text: str,
        language: str = "en",
    ) -> Tuple[torch.Tensor, int]:
        """Alias for synthesize_from_embeddings."""
        return self.synthesize_from_embeddings(
            text_embeddings, speaker_conditioning, text, language=language
        )

    @abstractmethod
    def synthesize(
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

    def synthesize_direct(
        self,
        text: str,
        speaker_conditioning: Dict[str, Any],
        language: str = "en",
        user_ref_text: Optional[str] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Backward-compatible alias for synthesize()."""
        return self.synthesize(
            text, speaker_conditioning, language=language, user_ref_text=user_ref_text
        )

