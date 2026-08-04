"""
Audio Artifact Detector for FlowEdit.

Implements combined gibberish detection combining:
- ASR target overlap
- Phoneme alignment score
- Repetition score
- Voiced ratio
- Duration ratio
- Spectral flatness
- Token diversity

Spectral entropy alone is NEVER used to reject speech.
"""

from dataclasses import dataclass
import numpy as np
import torch
import logging
from typing import Dict, Any, Optional

from flowedit.audio.audio_quality import compute_spectral_flatness, compute_silence_ratio, compute_clipping_ratio

logger = logging.getLogger(__name__)


@dataclass
class ArtifactAssessment:
    is_valid: bool
    gibberish_score: float               # Composite score 0.0 (clean) to 1.0 (severe gibberish)
    repetition_detected: bool
    truncation_detected: bool
    clipping_detected: bool
    excessive_silence_detected: bool
    rejection_reason: Optional[str] = None


class AudioArtifactDetector:
    """Evaluates synthesized waveforms for audio artifacts and gibberish."""

    def __init__(
        self,
        max_clipping_ratio: float = 0.10,
        max_silence_ratio: float = 0.60,
        min_voiced_ratio: float = 0.20,
    ):
        self.max_clipping_ratio = max_clipping_ratio
        self.max_silence_ratio = max_silence_ratio
        self.min_voiced_ratio = min_voiced_ratio

    def assess(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        planned_duration_sec: float,
        asr_target_overlap: float = 1.0,
        phoneme_alignment_score: float = 1.0,
        token_diversity: float = 1.0,
    ) -> ArtifactAssessment:
        if waveform.ndim > 1:
            wf = waveform.squeeze(0)
        else:
            wf = waveform

        actual_duration = wf.shape[-1] / float(sample_rate)
        clipping_ratio = compute_clipping_ratio(wf)
        silence_ratio = compute_silence_ratio(wf)
        spectral_flatness = compute_spectral_flatness(wf)

        # Voiced ratio: fraction of samples with non-trivial energy
        peak = torch.max(torch.abs(wf)).item()
        energy_thresh = max(1e-5, peak * 0.10)
        voiced_samples = torch.sum(torch.abs(wf) >= energy_thresh).item()
        voiced_ratio = voiced_samples / max(1, wf.numel())

        # Duration ratio: actual vs planned duration
        duration_ratio = actual_duration / max(0.1, planned_duration_sec)

        # Repetition score (0.0 to 1.0)
        repetition_score = max(0.0, 1.0 - token_diversity)

        # Truncation check
        truncation = duration_ratio < 0.65 or (actual_duration < 0.5 and planned_duration_sec > 1.5)

        # Clipping & silence checks
        clipping = clipping_ratio > self.max_clipping_ratio
        excessive_silence = silence_ratio > self.max_silence_ratio or voiced_ratio < self.min_voiced_ratio

        # Composite Gibberish Classifier combining all signals
        gibberish_score = (
            0.30 * (1.0 - max(0.0, min(1.0, asr_target_overlap)))
            + 0.25 * (1.0 - max(0.0, min(1.0, phoneme_alignment_score)))
            + 0.20 * repetition_score
            + 0.15 * max(0.0, 1.0 - duration_ratio)
            + 0.10 * max(0.0, spectral_flatness - 0.40)  # Extreme white-noise flatness penalty
        )
        gibberish_score = float(np.clip(gibberish_score, 0.0, 1.0))

        reasons = []
        if truncation:
            reasons.append(f"Truncated duration ({actual_duration:.2f}s vs planned {planned_duration_sec:.2f}s)")
        if repetition_score > 0.6:
            reasons.append("Token repetition loop detected")
        if clipping:
            reasons.append(f"Excessive clipping ({clipping_ratio:.2%})")
        if excessive_silence:
            reasons.append(f"Excessive silence / unvoiced signal (silence={silence_ratio:.2%})")
        if gibberish_score > 0.55:
            reasons.append(f"High composite gibberish score ({gibberish_score:.2f})")

        is_valid = len(reasons) == 0

        return ArtifactAssessment(
            is_valid=is_valid,
            gibberish_score=gibberish_score,
            repetition_detected=(repetition_score > 0.6),
            truncation_detected=truncation,
            clipping_detected=clipping,
            excessive_silence_detected=excessive_silence,
            rejection_reason=" | ".join(reasons) if reasons else None,
        )
