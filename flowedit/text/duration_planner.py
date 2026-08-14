"""
Duration and Pacing Planner for FlowEdit.

Derives natural speech duration and dynamic speed adjustment for F5-TTS
based on reference speech rate, target text syllables, punctuation pauses, and sentence padding.
"""

from dataclasses import dataclass
import logging
import re
from typing import Optional

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
        min_speech_rate: float = 2.5,
        max_speech_rate: float = 6.5,
    ):
        self.speed_floor = speed_floor
        self.speed_ceiling = speed_ceiling
        self.default_speed = default_speed
        self.sentence_end_padding_seconds = sentence_end_padding_seconds
        self.min_speech_rate = min_speech_rate
        self.max_speech_rate = max_speech_rate

    @staticmethod
    def estimate_syllables(text: str) -> int:
        """Estimate syllable count in text."""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return max(1, len(text) // 3)
        total = 0
        vowels = "aeiouy"
        for word in words:
            count = 0
            prev_vowel = False
            for ch in word:
                if ch in vowels:
                    if not prev_vowel:
                        count += 1
                    prev_vowel = True
                else:
                    prev_vowel = False
            if word.endswith("e") and count > 1:
                count -= 1
            total += max(1, count)
        return max(1, total)

    @staticmethod
    def calculate_punctuation_pauses(text: str) -> float:
        """Calculate total pause time added by commas, periods, etc."""
        pause = 0.0
        pause += len(re.findall(r'[,;:]', text)) * 0.15
        pause += len(re.findall(r'[.!?]', text)) * 0.30
        return pause

    def plan_duration(
        self,
        target_text: str,
        ref_audio_duration: Optional[float] = None,
        ref_text: Optional[str] = None,
        language: str = "en",
    ) -> PlannedDuration:
        """Compute dynamic target duration and dynamic speed factor."""
        target_syllables = self.estimate_syllables(target_text)
        punctuation_pauses = self.calculate_punctuation_pauses(target_text)

        if ref_audio_duration and ref_audio_duration > 0.5 and ref_text:
            ref_syllables = self.estimate_syllables(ref_text)
            raw_ref_rate = ref_syllables / float(ref_audio_duration)
            ref_rate = max(self.min_speech_rate, min(self.max_speech_rate, raw_ref_rate))
        else:
            ref_rate = 4.0

        raw_target_duration = target_syllables / ref_rate
        planned_duration = raw_target_duration + punctuation_pauses + self.sentence_end_padding_seconds

        unconstrained_duration = target_syllables / 4.0 + punctuation_pauses + 0.2
        if planned_duration > 0:
            raw_speed = unconstrained_duration / planned_duration
        else:
            raw_speed = self.default_speed

        dynamic_speed = max(self.speed_floor, min(self.speed_ceiling, raw_speed))

        return PlannedDuration(
            planned_duration_seconds=planned_duration,
            target_syllables=target_syllables,
            reference_speech_rate=ref_rate,
            punctuation_pause_seconds=punctuation_pauses,
            sentence_end_padding_seconds=self.sentence_end_padding_seconds,
            dynamic_speed_factor=dynamic_speed,
        )
