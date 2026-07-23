"""
FlowEdit: Lifelong Pronunciation Adaptation for TTS via Associative Memory

Adapts the FlowEdit framework (arXiv:2606.20518) to use XTTS-2 as the
frozen TTS backbone. Enables non-destructive pronunciation correction
through latent input optimization and Modern Hopfield Network memory.

Modules:
    backbone    - XTTS-2 frozen backbone wrapper
    alignment   - Whisper-based forced alignment (Stage 1)
    optimizer   - Latent input optimizer (Stage 2)
    memory      - Modern Hopfield Network (Stage 3)
    refiner     - Gated Hopfield retrieval at inference
    pipeline    - Correction loop & inference orchestrators
    utils       - Audio utilities, metrics
"""

__version__ = "0.1.0"
