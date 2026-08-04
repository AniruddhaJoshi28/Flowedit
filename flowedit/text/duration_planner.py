"""
Dynamic Duration and Pacing Planner for FlowEdit.

Derives natural speech duration and dynamic speed adjustment for F5-TTS
and other backbones based on reference voiced speech rate, target text
syllable/phoneme structure, punctuation pauses, and sentence padding.
"""

from dataclasses import dataclass
import logging
from typing import Optional

from flowedit.text.speech_input_processor import SpeechTextProcessor

logger = logging.getLogger(__name__)


@dataclass
class PlannedDuration:
    planned_duration_seconds: float
    target_syllables: int
    reference_speech_rate: float        # Syllables per second
    punctuation_pause_seconds: float
    sentence_end_padding_seconds: float
    dynamic_speed_factor: float          # Clamped dynamic speed (0.82 to 1.02)


class DurationPlanner:
    """Computes dynamic cadence and target duration for speech synthesis."""

    def __init__(
        self,
        speed_floor: float = 0.82,
        speed_ceiling: float = 1.02,
        default_speed: float = 0.90,
        sentence_end_padding_seconds: float = 0.35,
        min_speech_rate: float = 2.5,    # Minimum syllables per second
        max_speech_rate: float = 6.5,    # Maximum syllables per second
    ):
        self.speed_floor = speed_floor
        self.speed_ceiling = speed_ceiling
        self.default_speed = default_speed
        self.sentence_end_padding_seconds = sentence_end_padding_seconds
        self.min_speech_rate = min_speech_rate
        self.max_speech_rate = max_speech_rate

    def plan_duration(
        self,
        target_text: str,
        ref_audio_duration: Optional[float] = None,
        ref_text: Optional[str] = None,
        language: str = "en",
    ) -> PlannedDuration:
        """Computes dynamic target duration and dynamic speed factor.

        Formula:
            reference_rate = ref_syllables / max(0.5, ref_audio_duration)
            raw_target_duration = target_syllables / clamp(reference_rate, min_rate, max_rate)
            planned_duration = raw_target_duration + punctuation_pauses + sentence_end_padding
            dynamic_speed = clamp(default_duration / planned_duration, speed_floor, speed_ceiling)
        """
        target_syllables = SpeechTextProcessor.estimate_syllables(target_text)
        punctuation_pauses = SpeechTextProcessor._calculate_punctuation_pauses(target_text)

        # Calculate reference speech rate if reference audio and text are provided
        if ref_audio_duration and ref_audio_duration > 0.5 and ref_text:
            ref_syllables = SpeechTextProcessor.estimate_syllables(ref_text)
            raw_ref_rate = ref_syllables / float(ref_audio_duration)
            ref_rate = max(self.min_speech_rate, min(self.max_speech_rate, raw_ref_rate))
        else:
            ref_rate = 4.0  # Default ~4 syllables/sec for standard natural pacing

        raw_target_duration = target_syllables / ref_rate
        planned_duration = (
            raw_target_duration
            + punctuation_pauses
            + self.sentence_end_padding_seconds
        )

        # Dynamic speed calculation
        unconstrained_duration = target_syllables / 4.0 + punctuation_pauses + 0.2
        if planned_duration > 0:
            raw_speed = unconstrained_duration / planned_duration
        else:
            raw_speed = self.default_speed

        dynamic_speed = max(self.speed_floor, min(self.speed_ceiling, raw_speed))

        logger.info(
            f"Duration Planning: target_syllables={target_syllables}, ref_rate={ref_rate:.2f} syl/s, "
            f"planned_duration={planned_duration:.2f}s, dynamic_speed={dynamic_speed:.3f}"
        )

        return PlannedDuration(
            planned_duration_seconds=planned_duration,
            target_syllables=target_syllables,
            reference_speech_rate=ref_rate,
            punctuation_pause_seconds=punctuation_pauses,
            sentence_end_padding_seconds=self.sentence_end_padding_seconds,
            dynamic_speed_factor=dynamic_speed,
        )
