"""
Unit tests for the Modern Hopfield Memory and Refiner modules (arXiv:2606.20518).
"""

import torch
import torch.nn.functional as F
import pytest

from flowedit.config import MemoryConfig
from flowedit.memory.hopfield_memory import HopfieldMemory
from flowedit.refiner.hopfield_refiner import HopfieldRefiner


class TestHopfieldMemory:
    """Tests for the Modern Continuous Hopfield Associative Memory."""

    def setup_method(self):
        self.dim = 128
        self.config = MemoryConfig(
            max_entries=5,
            dedup_cosine_threshold=0.95,
            dedup_ema_decay=0.90,
            gate_threshold_init=2.0,
        )
        self.memory = HopfieldMemory(config=self.config, embedding_dim=self.dim)

    def test_empty_memory(self):
        assert self.memory.is_empty()
        assert self.memory.num_entries == 0

        query = torch.randn(1, 10, self.dim)
        res = self.memory.retrieve(query)
        assert res.retrieved_delta.shape == (1, 10, self.dim)
        assert not res.is_active

    def test_write_and_retrieve(self):
        key = F.normalize(torch.randn(self.dim), dim=0)
        value = torch.randn(self.dim)

        idx, action = self.memory.write(key, value, word="test")
        assert idx == 0
        assert action == "inserted"
        assert self.memory.num_entries == 1

        query = key.unsqueeze(0).unsqueeze(0)  # [1, 1, d]
        res = self.memory.retrieve(query)
        assert res.retrieved_delta.shape == (1, 1, self.dim)

    def test_deduplication(self):
        key = F.normalize(torch.randn(self.dim), dim=0)
        value1 = torch.randn(self.dim)
        value2 = torch.randn(self.dim)

        self.memory.write(key, value1, word="test")
        assert self.memory.num_entries == 1

        # Very similar key (cosine > 0.95)
        key_similar = key + 0.01 * torch.randn(self.dim)
        key_similar = F.normalize(key_similar, dim=0)
        idx, action = self.memory.write(key_similar, value2, word="test")

        assert action == "merged"
        assert self.memory.num_entries == 1

    def test_lru_pruning(self):
        # Fill capacity (max_entries=5)
        for i in range(5):
            k = F.normalize(torch.randn(self.dim), dim=0)
            v = torch.randn(self.dim)
            self.memory.write(k, v, word=f"w_{i}")

        assert self.memory.num_entries == 5

        # 6th insertion triggers LRU prune
        k_new = F.normalize(torch.randn(self.dim), dim=0)
        v_new = torch.randn(self.dim)
        self.memory.write(k_new, v_new, word="w_new")

        assert self.memory.num_entries == 5
        assert self.memory.entries[-1].word == "w_new"

    def test_context_key_computation(self):
        seq_len = 8
        emb = torch.randn(1, seq_len, self.dim)
        indices = [3, 4]
        key = self.memory.compute_context_key(emb, indices)

        assert key.shape == (self.dim,)
        assert torch.isclose(torch.norm(key), torch.tensor(1.0), atol=1e-4)


class TestHopfieldRefiner:
    """Tests for inference-time Hopfield Refiner."""

    def setup_method(self):
        self.dim = 128
        self.config = MemoryConfig(gate_threshold_init=1.0)
        self.memory = HopfieldMemory(config=self.config, embedding_dim=self.dim)
        self.refiner = HopfieldRefiner(memory=self.memory, config=self.config)

    def test_inactive_gate_pass_through(self):
        class DummyBackbone:
            embedding_dim = 128
            device = "cpu"
            def encode_text(self, text, lang="en"):
                return torch.randn(1, 6, 128)
            def synthesize_direct(self, text, speaker_conditioning, **kwargs):
                return torch.zeros(1, 24000), 24000
            def synthesize_from_embeddings(self, text_embeddings, speaker_conditioning, **kwargs):
                return torch.ones(1, 24000), 24000

        bb = DummyBackbone()
        res = self.refiner.forward(
            backbone=bb,
            text="hello world",
            speaker_conditioning={"audio_path": "fake.wav"},
        )
        assert not res.is_modified
        assert (res.waveform == 0).all()
