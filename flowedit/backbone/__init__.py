from .f5tts_wrapper import F5TTSBackbone
from .xtts_wrapper import XTTSBackbone


def create_backbone(config):
    """Factory function to create the correct backbone based on config.

    Args:
        config: BackboneConfig instance with backbone_type field.

    Returns:
        F5TTSBackbone or XTTSBackbone instance.

    Raises:
        ValueError: If backbone_type is not recognized.
    """
    backbone_type = getattr(config, "backbone_type", "xtts").lower()

    if backbone_type == "xtts":
        return XTTSBackbone(config)
    elif backbone_type in ("f5tts", "f5-tts", "f5"):
        return F5TTSBackbone(config)
    else:
        raise ValueError(
            f"Unknown backbone_type: '{backbone_type}'. "
            f"Supported: 'xtts', 'f5tts'"
        )
