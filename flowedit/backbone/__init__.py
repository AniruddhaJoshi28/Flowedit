from .base import TTSBackbone
from .f5tts_wrapper import F5TTSBackbone


def create_backbone(config=None) -> TTSBackbone:
    """Factory function to create the F5-TTS backbone.

    Args:
        config: BackboneConfig instance.

    Returns:
        F5TTSBackbone instance implementing TTSBackbone.
    """
    return F5TTSBackbone(config)


__all__ = ["TTSBackbone", "F5TTSBackbone", "create_backbone"]
