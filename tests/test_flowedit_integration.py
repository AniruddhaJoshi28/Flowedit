"""
Integration test for FlowEdit Modern Hopfield Memory Gating and Speaker Embedding.
"""

import pytest
import torch
import os
import wave
import struct
import math
from pathlib import Path

from flowedit.config import FlowEditConfig, BackboneConfig, MemoryConfig
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


def test_hopfield_gating_and_retrieval():
    dim = 128
    config = MemoryConfig(gate_threshold_init=0.0)
    memory = HopfieldMemory(config=config, embedding_dim=dim)

    # Base embeddings for "My name is Sahil Singh" (5 tokens)
    seq_len = 5
    torch.manual_seed(42)
    base_embeddings = torch.randn(1, seq_len, dim)

    # Suppose token 3 is target
    target_idx = 3
    key = base_embeddings[0, target_idx, :].clone()
    value = torch.ones(dim) * 0.5  # Perturbation vector delta*

    # Write correction to Hopfield memory
    memory.write(key=key, value=value, word="Sahil")

    # Run retrieval
    query = base_embeddings
    res = memory.retrieve(query)

    assert res.retrieved_delta.shape == (1, seq_len, dim)
    assert res.is_active


def test_backbone_speaker_embedding(sample_wav):
    config = BackboneConfig(device="cpu")
    backbone = F5TTSBackbone(config)
    
    # Test with sample audio and provided ref_text
    spk_info = backbone.get_speaker_embedding(sample_wav, ref_text="reference speech audio")
    assert os.path.exists(spk_info["processed_audio_path"])
    assert spk_info["text"] == "reference speech audio"

    from flowedit.audio.prompt_validator import ReferenceAudioError
    with pytest.raises(ReferenceAudioError):
        backbone.get_speaker_embedding(None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
