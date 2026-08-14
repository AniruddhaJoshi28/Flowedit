"""
Unit tests for AudioPreprocessor, PreparedAudio, prompt_validator, and ReferenceAudioError.
"""

import os
import wave
import struct
import tempfile
import pytest
import torch

from flowedit.audio.audio_preprocessor import AudioPreprocessor, PreparedAudio
from flowedit.audio.prompt_validator import validate_reference_audio, ReferenceAudioError
from flowedit.audio.audio_quality import compute_audio_quality_metrics


@pytest.fixture
def valid_wav_file():
    """Generates a temporary valid 16kHz PCM WAV audio file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sample_rate = 16000
        duration = 2.0
        n_samples = int(sample_rate * duration)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            for i in range(n_samples):
                t = i / float(sample_rate)
                # 440 Hz sine wave
                val = int(16000 * 0.5 * math_sin(2 * 3.14159265 * 440 * t))
                wf.writeframesraw(struct.pack("<h", val))
        tmp_path = tmp.name
    yield tmp_path
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def math_sin(x):
    import math
    return math.sin(x)


@pytest.fixture
def silent_wav_file():
    """Generates a nearly silent WAV file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sample_rate = 16000
        duration = 2.0
        n_samples = int(sample_rate * duration)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            for _ in range(n_samples):
                wf.writeframesraw(struct.pack("<h", 0))
        tmp_path = tmp.name
    yield tmp_path
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


@pytest.fixture
def empty_wav_file():
    """Generates an empty 0-byte file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name
    yield tmp_path
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_audio_preprocessor_valid(valid_wav_file):
    preprocessor = AudioPreprocessor(target_sample_rate=24000)
    prepared = preprocessor.preprocess(valid_wav_file)

    assert isinstance(prepared, PreparedAudio)
    assert prepared.sample_rate == 24000
    assert prepared.waveform.ndim == 2
    assert prepared.waveform.shape[0] == 1
    assert prepared.duration_seconds > 1.5
    assert prepared.rms > 0.01
    assert len(prepared.checksum) == 64


def test_prompt_validator_success(valid_wav_file):
    prepared = validate_reference_audio(valid_wav_file, min_duration_sec=0.5, max_duration_sec=10.0)
    assert prepared.duration_seconds >= 0.5


def test_prompt_validator_rejects_missing_file():
    with pytest.raises(ReferenceAudioError) as exc_info:
        validate_reference_audio("/nonexistent/path/to/audio.wav")
    assert "REFERENCE_AUDIO_" in exc_info.value.code


def test_prompt_validator_rejects_empty_file(empty_wav_file):
    with pytest.raises(ReferenceAudioError) as exc_info:
        validate_reference_audio(empty_wav_file)
    assert exc_info.value.code == "REFERENCE_AUDIO_EMPTY"


def test_prompt_validator_rejects_silent_file(silent_wav_file):
    with pytest.raises(ReferenceAudioError) as exc_info:
        validate_reference_audio(silent_wav_file)
    assert exc_info.value.code == "REFERENCE_AUDIO_SILENT"


def test_audio_quality_metrics(valid_wav_file):
    preprocessor = AudioPreprocessor(target_sample_rate=16000)
    prepared = preprocessor.preprocess(valid_wav_file)
    metrics = compute_audio_quality_metrics(prepared.waveform, prepared.sample_rate)

    assert "peak" in metrics
    assert "rms" in metrics
    assert "clipping_ratio" in metrics
    assert "silence_ratio" in metrics
    assert "spectral_flatness" in metrics
    assert metrics["rms"] > 0
