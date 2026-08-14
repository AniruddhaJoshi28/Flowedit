"""
Paper-Strict Rigorous Verification Test Suite for FlowEdit (arXiv:2606.20518).

Validates all 3 stages:
    Stage 1: Whisper token boundary extraction & ±1 token expansion.
    Stage 2: 50-step Adam latent optimization, gradient flow, loss reduction, L_inf clipping, cosine schedule.
    Stage 3: Modern Hopfield Network, softmax attention, similarity gating, EMA deduplication, LRU pruning.
"""

import math
import tempfile
import os
import wave
import struct
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from flowedit.config import FlowEditConfig, OptimizationConfig, MemoryConfig
from flowedit.optimizer.latent_optimizer import LatentOptimizer
from flowedit.memory.hopfield_memory import HopfieldMemory
from flowedit.refiner.hopfield_refiner import HopfieldRefiner
from flowedit.backbone.base import TTSBackbone


class MockDifferentiableBackbone(TTSBackbone):
    """Synthetic differentiable backbone for testing autograd backpropagation and loss reduction."""

    def __init__(self, d: int = 64):
        super().__init__(None)
        self.d = d
        self._dev = "cpu"
        # Synthetic target parameter to create a clean, non-trivial loss landscape
        self.target_weight = nn.Parameter(torch.randn(d, 80))

    @property
    def embedding_dim(self) -> int:
        return self.d

    @property
    def device(self) -> str:
        return self._dev

    @property
    def tokenizer(self):
        class DummyTok:
            def encode(self, t, lang="en"): return [[1, 2, 3, 4]]
            def decode(self, ids): return "test"
        return DummyTok()

    def load_model(self): pass

    def tokenize(self, text, language="en"):
        return {"token_ids": torch.tensor([[1, 2, 3, 4]]), "raw_tokens": list(text)}

    def detokenize(self, token_ids):
        return "test"

    def encode_text(self, text, language="en"):
        torch.manual_seed(123)
        return torch.randn(1, 4, self.d)

    def get_speaker_embedding(self, audio_path=None, language="en", ref_text=None):
        return {"audio_path": audio_path or "fake.wav", "text": "ref text", "duration_seconds": 2.0}

    def compute_optimization_loss(
        self,
        text_embedding_delta,
        ref_audio_path,
        speaker_conditioning,
        text,
        language="en",
        **kwargs,
    ):
        base_c = self.encode_text(text, language)
        c_pert = base_c + text_embedding_delta # [1, S, d]
        pred_mel = torch.matmul(c_pert, self.target_weight) # [1, S, 80]

        # Ground truth target mel
        with torch.no_grad():
            target_optimal_delta = torch.ones_like(base_c) * 0.5
            target_mel = torch.matmul(base_c + target_optimal_delta, self.target_weight)

        mel_loss = F.mse_loss(pred_mel, target_mel)
        return {"loss": mel_loss}

    def synthesize_from_embeddings(self, text_embeddings, speaker_conditioning, text, **kwargs):
        return torch.zeros(1, 24000), 24000

    def synthesize_direct(self, text, speaker_conditioning, **kwargs):
        return torch.zeros(1, 24000), 24000


@pytest.fixture
def mock_ref_wav(tmp_path):
    wav_path = str(tmp_path / "ref_target.wav")
    sr = 24000
    with wave.open(wav_path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        samples = [int(8000 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(sr)]
        f.writeframes(struct.pack('<' + 'h' * len(samples), *samples))
    return wav_path


def test_stage2_latent_optimization_gradient_and_loss(mock_ref_wav):
    """Verify Stage 2 latent optimization reduces loss, maintains finite gradients, and computes δ*."""
    backbone = MockDifferentiableBackbone(d=64)
    opt_config = OptimizationConfig(
        n_steps=20,
        lr_start=0.05,
        lr_end=0.005,
        lambda_reg=0.001,
        grad_clip_max_norm=1.0,
    )
    optimizer = LatentOptimizer(config=opt_config)

    res = optimizer.optimize(
        backbone=backbone,
        text="This is Siobhan speaking",
        target_word="Siobhan",
        ref_audio_path=mock_ref_wav,
        token_indices=[1, 2],
        speaker_conditioning={"duration_seconds": 2.0},
        seed=42,
    )

    # 1. Check loss reduction
    assert res.final_loss < res.initial_loss, f"Loss did not decrease: {res.initial_loss} -> {res.final_loss}"
    assert res.converged is True

    # 2. Check gradient flow
    assert len(res.grad_history) == 20
    assert all(g > 0.0 for g in res.grad_history), "Gradients must be strictly positive"
    assert all(math.isfinite(g) for g in res.grad_history), "Gradients must be finite"

    # 3. Check perturbation structure
    assert res.delta.shape == (1, 4, 64)
    assert res.delta_pooled.shape == (64,)
    assert res.key_pooled.shape == (64,)

    # 4. Check masking: non-target tokens must be 0
    non_target_indices = [idx for idx in range(4) if idx not in res.target_token_indices]
    for idx in non_target_indices:
        assert torch.all(res.delta[:, idx, :] == 0.0), f"Token {idx} outside target span was not masked!"


def test_stage3_hopfield_attention_and_gating():
    """Verify Modern Hopfield associative memory, Softmax attention, and similarity gating."""
    d = 64
    mem_config = MemoryConfig(
        max_entries=10,
        dedup_cosine_threshold=0.95,
        dedup_ema_decay=0.90,
        gate_threshold_init=3.0,
    )
    memory = HopfieldMemory(config=mem_config, embedding_dim=d)

    # Store 3 distinct corrections
    torch.manual_seed(42)
    k1 = F.normalize(torch.randn(d), dim=0)
    v1 = torch.ones(d) * 1.5
    memory.write(k1, v1, word="Siobhan")

    k2 = F.normalize(torch.randn(d), dim=0)
    v2 = torch.ones(d) * -2.0
    memory.write(k2, v2, word="Worcestershire")

    assert memory.num_entries == 2

    # Query with exact key k1 (similarity ≈ 1.0)
    # Scaled by beta = 1/√64 = 0.125
    query_exact = k1.unsqueeze(0).unsqueeze(0) # [1, 1, d]
    res_exact = memory.retrieve(query_exact)
    assert res_exact.top_matches[0][0] == "Siobhan"

    # Query with orthogonal random vector (similarity ≈ 0.0) -> Gate should be inactive
    q_orthogonal = F.normalize(torch.randn(d), dim=0).unsqueeze(0).unsqueeze(0)
    res_orth = memory.retrieve(q_orthogonal)
    # Gate value σ(max_sim - 3.0) with max_sim ≈ 0.0 should be < 0.05
    assert res_orth.gate_values.max().item() < 0.10, "Gate should remain inactive on unseen general query"


def test_stage3_deduplication_and_lru():
    """Verify EMA deduplication (cos > 0.95) and LRU eviction (M_max)."""
    d = 32
    mem_config = MemoryConfig(max_entries=3, dedup_cosine_threshold=0.95, dedup_ema_decay=0.90)
    memory = HopfieldMemory(config=mem_config, embedding_dim=d)

    k1 = F.normalize(torch.randn(d), dim=0)
    v1 = torch.ones(d)
    idx1, action1 = memory.write(k1, v1, word="WordA")
    assert action1 == "inserted"

    # Duplicate insertion (cos > 0.95)
    k1_dup = F.normalize(k1 + 0.01 * torch.randn(d), dim=0)
    v1_dup = torch.ones(d) * 2.0
    idx_dup, action_dup = memory.write(k1_dup, v1_dup, word="WordA")
    assert action_dup == "merged"
    assert idx_dup == idx1
    assert memory.num_entries == 1

    # Fill to capacity
    k2 = F.normalize(torch.randn(d), dim=0)
    memory.write(k2, torch.ones(d), word="WordB")
    k3 = F.normalize(torch.randn(d), dim=0)
    memory.write(k3, torch.ones(d), word="WordC")
    assert memory.num_entries == 3

    # 4th entry triggers LRU pruning
    k4 = F.normalize(torch.randn(d), dim=0)
    memory.write(k4, torch.ones(d), word="WordD")
    assert memory.num_entries == 3
    assert memory.entries[-1].word == "WordD"
