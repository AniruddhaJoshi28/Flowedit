"""
Unit tests for the Modern Hopfield Memory and Refiner modules (arXiv:2606.20518).
Tests cluster averaging, contextual homograph disambiguation, multi-word retrieval, and cross-sentence transfer.
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
            dedup_cosine_threshold=0.72,
            homograph_sim_threshold=0.72,
            dedup_ema_decay=0.90,
            gate_threshold_init=7.0,
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

    def test_cluster_averaging(self):
        """Verify that multiple corrections for the same word in matching context are averaged."""
        key = F.normalize(torch.randn(self.dim), dim=0)
        value1 = torch.ones(self.dim) * 2.0
        word_delta1 = torch.ones(1, 5, self.dim) * 1.0

        idx1, action1 = self.memory.write(
            key=key,
            value=value1,
            word="pipes",
            carrier_text="The pipes are made up of lead",
            word_delta=word_delta1,
        )
        assert action1 == "inserted"
        assert self.memory.num_entries == 1

        # Second correction for same word with similar context (cosine > 0.72)
        key2 = F.normalize(key + 0.05 * torch.randn(self.dim), dim=0)
        value2 = torch.ones(self.dim) * 4.0
        word_delta2 = torch.ones(1, 5, self.dim) * 3.0

        idx2, action2 = self.memory.write(
            key=key2,
            value=value2,
            word="pipes",
            carrier_text="These copper pipes are long",
            word_delta=word_delta2,
        )

        assert action2 == "averaged"
        assert idx2 == idx1
        assert self.memory.num_entries == 1
        entry = self.memory.entries[0]
        assert entry.access_count == 2
        # Value should be average: (2.0 + 4.0)/2 = 3.0
        assert torch.allclose(entry.value, torch.ones(self.dim) * 3.0, atol=1e-4)
        # word_delta should be average: (1.0 + 3.0)/2 = 2.0
        assert torch.allclose(entry.word_delta, torch.ones(1, 5, self.dim) * 2.0, atol=1e-4)
        assert len(entry.carrier_texts) == 2

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
        seq_len = 24
        emb = torch.randn(1, seq_len, self.dim)
        indices = [10, 11, 12, 13]  # target word in middle
        key = self.memory.compute_context_key(emb, indices)

        assert key.shape == (self.dim,)
        assert torch.isclose(torch.norm(key), torch.tensor(1.0), atol=1e-4)

    def test_contextual_homograph_storage_and_disambiguation(self):
        """Verify that identical words with distinct contexts are stored as separate entries and disambiguated."""
        torch.manual_seed(42)
        seq_len = 20
        emb_metal = torch.randn(1, seq_len, self.dim)
        lead_indices_a = [16, 17, 18, 19]  # "lead" (metal)
        key_metal = self.memory.compute_context_key(emb_metal, lead_indices_a)
        val_metal = torch.ones(self.dim) * 1.5
        w_delta_metal = torch.ones(1, 4, self.dim) * 2.0

        idx1, action1 = self.memory.write(
            key=key_metal,
            value=val_metal,
            word="lead",
            carrier_text="The pipe was made of lead",
            token_indices=lead_indices_a,
            word_delta=w_delta_metal,
        )
        assert action1 == "inserted"
        assert self.memory.num_entries == 1

        # Simulate Sentence B: "She will lead the team" (orthogonal context)
        emb_leader = torch.randn(1, seq_len, self.dim)
        lead_indices_b = [9, 10, 11, 12]
        key_leader = self.memory.compute_context_key(emb_leader, lead_indices_b)
        val_leader = torch.ones(self.dim) * -1.5
        w_delta_leader = torch.ones(1, 4, self.dim) * -2.0

        idx2, action2 = self.memory.write(
            key=key_leader,
            value=val_leader,
            word="lead",
            carrier_text="She will lead the team",
            token_indices=lead_indices_b,
            word_delta=w_delta_leader,
        )
        # Must be stored as a distinct contextual homograph entry!
        assert action2 == "inserted"
        assert self.memory.num_entries == 2
        assert idx1 != idx2

        # Retrieve on query matching metal context
        query_metal = emb_metal
        res_metal = self.memory.retrieve(query_metal, text="The pipe was made of lead")
        assert res_metal.top_matches[0][0] == "lead"
        assert res_metal.matched_carrier_text == "The pipe was made of lead"

        # Retrieve on query matching leader context
        query_leader = emb_leader
        res_leader = self.memory.retrieve(query_leader, text="She will lead the team")
        assert res_leader.top_matches[0][0] == "lead"
        assert res_leader.matched_carrier_text == "She will lead the team"

    def test_multi_word_simultaneous_retrieval(self):
        """Verify that a sentence containing multiple corrected words activates both corrections."""
        torch.manual_seed(42)
        text = "The pipes are made of lead"
        seq_len = len(text)
        emb = torch.randn(1, seq_len, self.dim)

        # Entry 1: pipes at [4:9]
        pipes_indices = list(range(4, 9))
        k_pipes = self.memory.compute_context_key(emb, pipes_indices)
        v_pipes = torch.ones(self.dim) * 1.0
        w_pipes = torch.ones(1, 5, self.dim) * 1.5
        self.memory.write(k_pipes, v_pipes, word="pipes", carrier_text=text, word_delta=w_pipes)

        # Entry 2: lead at [22:26]
        lead_indices = list(range(22, 26))
        k_lead = self.memory.compute_context_key(emb, lead_indices)
        v_lead = torch.ones(self.dim) * 2.0
        w_lead = torch.ones(1, 4, self.dim) * 2.5
        self.memory.write(k_lead, v_lead, word="lead", carrier_text=text, word_delta=w_lead)

        assert self.memory.num_entries == 2

        # Retrieve with the full sentence
        res = self.memory.retrieve(emb, text=text)
        assert res.is_active
        words_found = [span.word for span in res.matched_spans]
        assert "pipes" in words_found
        assert "lead" in words_found
        # Verify delta is injected at both word positions
        assert res.retrieved_delta[0, 4:9, :].abs().sum() > 0
        assert res.retrieved_delta[0, 22:26, :].abs().sum() > 0


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
                return torch.randn(1, len(text), 128)
            def synthesize_direct(self, text, speaker_conditioning, **kwargs):
                delta = kwargs.get("text_embedding_delta", None)
                if delta is not None:
                    return torch.ones(1, 24000), 24000
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
