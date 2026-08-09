from .base import TTSBackbone, OptimizationMode
from .f5tts_wrapper import F5TTSBackbone


def create_backbone(config) -> TTSBackbone:
    """Factory function to create the correct backbone based on config.

    Args:
        config: BackboneConfig instance.

    Returns:
        F5TTSBackbone instance implementing TTSBackbone.
    """
    return F5TTSBackbone(config)

