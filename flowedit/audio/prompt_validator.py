"""
Shared Reference Audio Validator for FlowEdit.

Enforces strict validation rules on reference speaker audio.
Raises ReferenceAudioError when validation fails. NO SILENT FALLBACK.
"""

import os
import logging
from typing import Optional
from flowedit.audio.audio_preprocessor import PreparedAudio, AudioPreprocessor

logger = logging.getLogger(__name__)


class ReferenceAudioError(Exception):
    """Raised when user-provided reference audio fails validation."""

    def __init__(self, message: str, code: str = "REFERENCE_AUDIO_INVALID"):
        super().__init__(message)
        self.message = message
        self.code = code


def validate_reference_audio(
    file_path: Optional[str],
    min_duration_sec: float = 0.5,
    max_duration_sec: float = 30.0,
    min_rms_energy: float = 1e-4,
    max_clipping_ratio: float = 0.15,
    target_sample_rate: int = 24000,
) -> PreparedAudio:
    """Validates user-supplied reference speaker audio.

    Raises ReferenceAudioError if:
    - Audio path is None, empty, or file does not exist.
    - File size is 0 bytes or unreadable.
    - Decoding fails.
    - Audio is shorter than min_duration_sec or longer than max_duration_sec.
    - Audio is nearly silent (RMS energy < min_rms_energy).
    - Audio is severely clipped (clipping_ratio > max_clipping_ratio).

    NEVER performs silent fallback to default demo audio.
    """
    if not file_path:
        raise ReferenceAudioError(
            "Reference audio file path is required but was not provided.",
            code="REFERENCE_AUDIO_MISSING",
        )

    if not os.path.exists(file_path):
        raise ReferenceAudioError(
            f"Reference audio file '{file_path}' does not exist.",
            code="REFERENCE_AUDIO_NOT_FOUND",
        )

    if os.path.getsize(file_path) == 0:
        raise ReferenceAudioError(
            f"Reference audio file '{file_path}' is empty (0 bytes).",
            code="REFERENCE_AUDIO_EMPTY",
        )

    preprocessor = AudioPreprocessor(target_sample_rate=target_sample_rate)
    try:
        prepared = preprocessor.preprocess(file_path, target_sr=target_sample_rate)
    except Exception as e:
        raise ReferenceAudioError(
            f"Corrupt or unreadable reference audio file '{file_path}': {e}",
            code="REFERENCE_AUDIO_CORRUPT",
        ) from e

    # Validation Checks
    if prepared.duration_seconds < min_duration_sec:
        raise ReferenceAudioError(
            f"Reference audio duration ({prepared.duration_seconds:.2f}s) is shorter than minimum allowed ({min_duration_sec}s).",
            code="REFERENCE_AUDIO_TOO_SHORT",
        )

    if prepared.duration_seconds > max_duration_sec:
        raise ReferenceAudioError(
            f"Reference audio duration ({prepared.duration_seconds:.2f}s) exceeds maximum configured limit ({max_duration_sec}s).",
            code="REFERENCE_AUDIO_TOO_LONG",
        )

    if prepared.rms < min_rms_energy:
        raise ReferenceAudioError(
            f"Reference audio is nearly silent (RMS energy {prepared.rms:.6f} < threshold {min_rms_energy}).",
            code="REFERENCE_AUDIO_SILENT",
        )

    if prepared.clipping_ratio > max_clipping_ratio:
        raise ReferenceAudioError(
            f"Reference audio is severely clipped (clipping ratio {prepared.clipping_ratio:.2%} > limit {max_clipping_ratio:.2%}).",
            code="REFERENCE_AUDIO_CLIPPED",
        )

    logger.info(
        f"Reference audio '{file_path}' validated successfully: "
        f"duration={prepared.duration_seconds:.2f}s, sr={prepared.sample_rate}, "
        f"rms={prepared.rms:.4f}, clipping={prepared.clipping_ratio:.2%}, checksum={prepared.checksum[:8]}"
    )
    return prepared
