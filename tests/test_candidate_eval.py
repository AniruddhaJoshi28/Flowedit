"""
Unit tests for AudioArtifactDetector and CandidateSelector.
"""

import pytest
import torch
import numpy as np

from flowedit.evaluation.audio_artifact_detector import AudioArtifactDetector
from flowedit.evaluation.candidate_selector import CandidateSelector


def test_audio_artifact_detector_clean_waveform():
    detector = AudioArtifactDetector()
    sample_rate = 24000
    duration = 2.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    # Standard clean 440Hz sine wave
    waveform = 0.5 * torch.sin(2 * 3.14159265 * 440 * t).unsqueeze(0)

    assessment = detector.assess(
        waveform=waveform,
        sample_rate=sample_rate,
        planned_duration_sec=2.0,
    )

    assert assessment.is_valid is True
    assert assessment.gibberish_score < 0.40
    assert assessment.truncation_detected is False
    assert assessment.clipping_detected is False


def test_audio_artifact_detector_clipped_waveform():
    detector = AudioArtifactDetector(max_clipping_ratio=0.10)
    sample_rate = 24000
    waveform = torch.ones(1, sample_rate * 2)  # Severe clipping (all 1.0)

    assessment = detector.assess(
        waveform=waveform,
        sample_rate=sample_rate,
        planned_duration_sec=2.0,
    )

    assert assessment.clipping_detected is True
    assert assessment.is_valid is False


def test_candidate_selector_staged_fast_path():
    selector = CandidateSelector()
    sample_rate = 24000
    t = torch.linspace(0, 2.0, sample_rate * 2)
    clean_wav = 0.5 * torch.sin(2 * 3.14159265 * 440 * t).unsqueeze(0)

    call_count = 0

    def mock_synthesis(idx):
        nonlocal call_count
        call_count += 1
        return clean_wav, sample_rate

    wav, sr, eval_obj = selector.select_staged_candidate(
        synthesis_fn=mock_synthesis,
        planned_duration_sec=2.0,
        run_slow_path=False,
    )

    # Fast path passed -> should return after Candidate 1 (call_count == 1)
    assert call_count == 1
    assert eval_obj.fast_path_passed is True
    assert sr == sample_rate
