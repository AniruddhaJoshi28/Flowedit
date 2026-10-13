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


def test_homograph_sense_alignment_with_carrier_variation():
    """Verify that homograph 'lead' matches the metal occurrence ('led') and preserves the leader occurrence."""
    dim = 128
    config = MemoryConfig(gate_threshold_init=0.0)
    memory = HopfieldMemory(config=config, embedding_dim=dim)

    # Register 'lead' (metal sense) with carrier text "This toy is made up of lead"
    key = torch.randn(dim)
    key = torch.nn.functional.normalize(key, p=2, dim=-1)
    val = torch.ones(dim) * 0.4
    memory.write(
        key=key,
        value=val,
        word="lead",
        carrier_text="This toy is made up of lead",
        phonetic_text="led",
    )

    # Test sentence with slight typo/carrier variation: "The toy is made up lead and Anurag is the lead of data science"
    test_text = "The toy is made up lead and Anurag is the lead of data science"
    query_emb = torch.randn(1, len(test_text.split()), dim)

    retrieval = memory.retrieve(query_emb, text=test_text)
    assert retrieval.is_active, "Memory should be active for the metal sense of lead"
    assert len(retrieval.matched_spans) == 1, "Only the metal occurrence of lead should match"
    matched = retrieval.matched_spans[0]
    assert matched.word == "lead"
    assert matched.phonetic_text == "led"
    # Occurrence 1 character span starts around character 19
    assert matched.char_start == 19


def test_f5tts_seed_and_parameter_handling():
    """Verify that F5TTSBackbone accepts seed and sets reproducible manual seed."""
    config = BackboneConfig(device="cpu")
    backbone = F5TTSBackbone(config)
    
    # Check that synthesize_direct accepts seed
    spk_info = {"audio_path": "", "processed_audio_path": "", "text": "."}
    wav1, sr1 = backbone.synthesize_direct("Hello world", spk_info, seed=42)
    wav2, sr2 = backbone.synthesize_direct("Hello world", spk_info, seed=42)
    assert wav1.shape == wav2.shape
    assert sr1 == sr2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

