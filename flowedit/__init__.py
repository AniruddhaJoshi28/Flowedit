"""
FlowEdit: Lifelong Pronunciation Adaptation for Flow-Matching TTS via Associative Memory.

Paper Reference: arXiv:2606.20518.
Continuous Flow-Matching (F5-TTS DiT) TTS Backbone with Modern Hopfield Associative Memory.

Modules:
    backbone    - F5-TTS frozen Flow-Matching DiT backbone wrapper
    alignment   - Whisper forced alignment (Stage 1)
    optimizer   - Latent input optimizer (Stage 2)
    memory      - Modern Hopfield Network associative memory (Stage 3)
    refiner     - Similarity-gated Hopfield retrieval at inference
    pipeline    - Correction loop & inference orchestrators
    utils       - Audio utilities, metrics
"""

from .config import FlowEditConfig
from .backbone import create_backbone, F5TTSBackbone
from .memory import HopfieldMemory
from .refiner import HopfieldRefiner
from .pipeline import CorrectionLoop, FlowEditInference

__version__ = "0.2.0"
