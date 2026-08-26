"""
Abstract Flow-Matching TTS Backbone Interface for FlowEdit.

Paper Reference: FlowEdit (arXiv:2606.20518), Section 3.1 & 3.2.
All backbones implement this abstract base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn


class TTSBackbone(nn.Module, ABC):
    """Abstract Base Class for FlowEdit TTS Backbones."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimension d of the text embedding space (Paper: d = 1024 or 512)."""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Return the active device (cuda/cpu)."""
        pass

    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        """Return the tokenizer instance for token-text mapping."""
        pass

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights and initialize internal neural modules."""
        pass

    @abstractmethod
    def tokenize(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Tokenize text string into token dictionary containing 'token_ids'."""
        pass

    @abstractmethod
    def detokenize(self, token_ids: torch.Tensor) -> str:
        """Convert token IDs back into text representation."""
        pass

    def get_token_ids(self, text: str, language: str = "en") -> torch.Tensor:
        """Tokenize text string into token IDs [1, S]."""
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
        text_embedding_delta: torch.Tensor,
        ref_audio_path: str,
        speaker_conditioning: Dict[str, Any],
        text: str,
        language: str = "en",
        target_word_start_sample: Optional[int] = None,
        target_word_end_sample: Optional[int] = None,
        ref_start_time: Optional[float] = None,
        ref_end_time: Optional[float] = None,
        seed: Optional[int] = 42,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute differentiable loss for optimizing perturbation δ (Paper Eq. 3).
        
        Returns:
            Dict containing at least:
                - "loss": scalar loss tensor for backpropagation.
        """
        pass

    @abstractmethod
    def synthesize_from_embeddings(
        self,
        text_embeddings: torch.Tensor,
        speaker_conditioning: Dict[str, Any],
        text: str,
        language: str = "en",
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """Synthesize audio using (possibly perturbed/refined) text embeddings c + δ.
        
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
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """Direct synthesis without embedding modifications (when Hopfield gate is inactive).
        
        Returns:
            Tuple of (waveform tensor [1, T], sample_rate int)
        """
        pass

    def synthesize(
        self,
        text: str,
        speaker_conditioning: Dict[str, Any],
        language: str = "en",
        user_ref_text: Optional[str] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """Alias for synthesize_direct."""
        return self.synthesize_direct(
            text, speaker_conditioning, language=language, user_ref_text=user_ref_text, **kwargs
        )

    def synthesize_baseline(
        self,
        text: str,
        speaker_conditioning: Dict[str, Any],
        language: str = "en",
        user_ref_text: Optional[str] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """Synthesize pure baseline audio using raw base model without FlowEdit memory."""
        return self.synthesize_direct(
            text, speaker_conditioning, language=language, user_ref_text=user_ref_text, **kwargs
        )

