"""
Integration test for FlowEdit hook offset and Hopfield Memory Gating.
"""

import pytest
import torch
import os
import wave
import struct
import math
from pathlib import Path

from flowedit.config import FlowEditConfig, BackboneConfig
from flowedit.memory.hopfield_memory import HopfieldMemory
from flowedit.refiner.hopfield_refiner import HopfieldRefiner
from flowedit.backbone.f5tts_wrapper import F5TTSBackbone


@pytest.fixture
def sample_wav(tmp_path):
    wav_path = str(tmp_path / "test_audio.wav")
    sr = 24000
    duration = 1.0
    num_samples = int(sr * duration)
    with wave.open(wav_path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        samples = [int(16000 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(num_samples)]
        f.writeframes(struct.pack('<' + 'h' * len(samples), *samples))
    return wav_path


def test_hopfield_gating_and_refinement():
    dim = 512
    memory = HopfieldMemory(dim=dim)
    refiner = HopfieldRefiner(memory)

    # Base embeddings for "My name is Sahil Singh Rangra" (say 6 tokens)
    seq_len = 6
    torch.manual_seed(42)
    base_embeddings = torch.randn(1, seq_len, dim)

    # Suppose token 3 ("Sahil") is target
    target_idx = 3
    target_embed = base_embeddings[0, target_idx, :]
    key = target_embed.clone()
    value = torch.ones(dim) * 0.5  # Perturbation vector delta*

    # Write correction to Hopfield memory
    memory.write(key=key, value=value, word="Sahil")

    # Run refiner on text embeddings
    refined_embeddings, gate_values = refiner(base_embeddings)

    # Gate value at target index 3 should be active (> 0.5)
    assert gate_values[0, target_idx].item() > 0.5, f"Expected active gate at index {target_idx}, got {gate_values[0, target_idx].item()}"

    # Non-target indices should be near 0.0
    for i in range(seq_len):
        if i != target_idx:
            assert gate_values[0, i].item() < 0.5, f"Expected inactive gate at index {i}, got {gate_values[0, i].item()}"

    # Difference in target embeddings should equal scale * gate * value
    scale = getattr(refiner.config, "perturbation_scale", 1.8)
    diff = refined_embeddings[0, target_idx, :] - base_embeddings[0, target_idx, :]
    expected_diff = scale * gate_values[0, target_idx] * value
    assert torch.allclose(diff, expected_diff, atol=1e-4)


def test_backbone_speaker_embedding(sample_wav):
    config = BackboneConfig(device="cpu")
    backbone = F5TTSBackbone(config)
    
    # Test with sample audio
    spk_info = backbone.get_speaker_embedding(sample_wav)
    assert os.path.exists(spk_info["processed_audio_path"])
    assert "text" in spk_info

    # Test with None (fallback)
    spk_info_fallback = backbone.get_speaker_embedding(None)
    assert os.path.exists(spk_info_fallback["processed_audio_path"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
