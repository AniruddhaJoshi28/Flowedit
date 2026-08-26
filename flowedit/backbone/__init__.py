from .base import TTSBackbone
from .xtts_wrapper import XTTSBackbone
from .f5tts_wrapper import F5TTSBackbone


def create_backbone(config=None) -> TTSBackbone:
    """Factory function to create the TTS backbone (XTTS-v2 by default).

    Args:
        config: BackboneConfig instance.

    Returns:
        TTSBackbone instance (XTTSBackbone or F5TTSBackbone).
    """
    if config is not None:
        btype = getattr(config, "backbone_type", "xtts").lower()
        if btype in ("f5tts", "f5-tts"):
            return F5TTSBackbone(config)
    return XTTSBackbone(config)


__all__ = ["TTSBackbone", "XTTSBackbone", "F5TTSBackbone", "create_backbone"]
