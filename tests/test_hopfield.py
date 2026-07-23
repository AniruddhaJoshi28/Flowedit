"""
Unit tests for the Hopfield Memory and Refiner modules.

These test the core FlowEdit memory system without requiring
the full TTS backbone (which needs GPU and model downloads).
"""

import torch
import torch.nn.functional as F
import pytest
import tempfile
import os

from flowedit.config import MemoryConfig
from flowedit.memory.hopfield_memory import HopfieldMemory
from flowedit.refiner.hopfield_refiner import HopfieldRefiner


class TestHopfieldMemory:
    """Tests for the Modern Hopfield Network memory."""

    def setup_method(self):
        """Create a fresh memory for each test."""
        self.dim = 128
        self.config = MemoryConfig(max_entries=10, dedup_cosine_threshold=0.95)
        self.memory = HopfieldMemory(dim=self.dim, config=self.config)

    def test_empty_memory(self):
        """Empty memory should return zeros on retrieval."""
        assert self.memory.is_empty
        assert self.memory.size == 0

        query = torch.randn(self.dim)
        retrieved, sim = self.memory.retrieve(query)

        assert retrieved.shape == (self.dim,)
        assert sim.item() == -1.0

    def test_write_and_retrieve(self):
        """Basic write then retrieve should work."""
        key = F.normalize(torch.randn(self.dim), dim=0)
        value = torch.randn(self.dim)

        idx = self.memory.write(key, value, word="test")
        assert idx == 0
        assert self.memory.size == 1

        # Retrieve with the same key
        retrieved, sim = self.memory.retrieve(key)
        assert retrieved.shape == (self.dim,)
        # With only one entry, retrieval should be close to the stored value
        cosine = F.cosine_similarity(retrieved.unsqueeze(0), value.unsqueeze(0))
        assert cosine.item() > 0.9

    def test_multiple_writes(self):
        """Multiple writes should all be stored."""
        for i in range(5):
            key = F.normalize(torch.randn(self.dim), dim=0)
            value = torch.randn(self.dim)
            self.memory.write(key, value, word=f"word_{i}")

        assert self.memory.size == 5

    def test_deduplication(self):
        """Writing a very similar key should EMA-update, not insert."""
        key = F.normalize(torch.randn(self.dim), dim=0)
        value1 = torch.randn(self.dim)
        value2 = torch.randn(self.dim)

        self.memory.write(key, value1, word="test")
        assert self.memory.size == 1

        # Write almost identical key (cosine > 0.95)
        key_similar = key + 0.01 * torch.randn(self.dim)
        key_similar = F.normalize(key_similar, dim=0)
        self.memory.write(key_similar, value2, word="test")

        # Should still be 1 (EMA update, not new insertion)
        assert self.memory.size == 1

    def test_lru_pruning(self):
        """Memory should prune LRU entries when at capacity."""
        # Fill to capacity
        for i in range(10):
            key = F.normalize(torch.randn(self.dim), dim=0)
            value = torch.randn(self.dim)
            self.memory.write(key, value, word=f"word_{i}")

        assert self.memory.size == 10

        # One more should trigger pruning
        key = F.normalize(torch.randn(self.dim), dim=0)
        value = torch.randn(self.dim)
        self.memory.write(key, value, word="word_new")

        assert self.memory.size == 10  # Still at capacity

    def test_save_and_load(self):
        """Memory should persist correctly to disk."""
        # Add some corrections
        for i in range(3):
            key = F.normalize(torch.randn(self.dim), dim=0)
            value = torch.randn(self.dim)
            self.memory.write(key, value, word=f"word_{i}")

        # Save
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        self.memory.save(path)

        # Load into new memory
        new_memory = HopfieldMemory(dim=self.dim, config=self.config)
        new_memory.load(path)

        assert new_memory.size == 3

        # Verify retrieval works the same
        query = torch.randn(self.dim)
        r1, s1 = self.memory.retrieve(query)
        r2, s2 = new_memory.retrieve(query)

        assert torch.allclose(r1, r2, atol=1e-5)

        # Cleanup
        os.unlink(path)

    def test_batch_retrieval(self):
        """Batch retrieval should work for sequence of queries."""
        # Store a few corrections
        for i in range(3):
            key = F.normalize(torch.randn(self.dim), dim=0)
            value = torch.randn(self.dim)
            self.memory.write(key, value, word=f"word_{i}")

        # Batch query
        queries = torch.randn(10, self.dim)
        retrieved, sims = self.memory.retrieve_batch(queries)

        assert retrieved.shape == (10, self.dim)
        assert sims.shape == (10,)

    def test_list_corrections(self):
        """Listing corrections should return metadata."""
        self.memory.write(
            F.normalize(torch.randn(self.dim), dim=0),
            torch.randn(self.dim),
            word="Siobhan",
        )
        self.memory.write(
            F.normalize(torch.randn(self.dim), dim=0),
            torch.randn(self.dim),
            word="Nguyen",
        )

        corrections = self.memory.list_corrections()
        assert len(corrections) == 2
        assert corrections[0]["word"] == "Siobhan"
        assert corrections[1]["word"] == "Nguyen"

    def test_clear(self):
        """Clear should remove all corrections."""
        self.memory.write(
            F.normalize(torch.randn(self.dim), dim=0),
            torch.randn(self.dim),
            word="test",
        )
        assert self.memory.size == 1

        self.memory.clear()
        assert self.memory.size == 0
        assert self.memory.is_empty


class TestHopfieldRefiner:
    """Tests for the gated retrieval refiner."""

    def setup_method(self):
        self.dim = 128
        self.config = MemoryConfig(gate_threshold_init=5.0)
        self.memory = HopfieldMemory(dim=self.dim, config=self.config)
        self.refiner = HopfieldRefiner(memory=self.memory, config=self.config)

    def test_empty_memory_passthrough(self):
        """With empty memory, refiner should be pure passthrough."""
        embeddings = torch.randn(1, 10, self.dim)
        refined, gate = self.refiner(embeddings)

        # Should be identical (zero forgetting guarantee)
        assert torch.allclose(refined, embeddings)
        assert (gate == 0).all()

    def test_gate_suppression(self):
        """Gate should suppress irrelevant retrievals."""
        # Store a correction for a specific embedding pattern
        key = F.normalize(torch.randn(self.dim), dim=0)
        value = torch.randn(self.dim) * 0.1
        self.memory.write(key, value, word="test")

        # Query with completely different embeddings
        random_embeddings = torch.randn(1, 5, self.dim)
        refined, gate = self.refiner(random_embeddings)

        # Gate should be near-zero for unrelated queries
        # (high τ = 5.0 means similarity must be very high to activate)
        # The max gate value should be very small
        assert gate.max().item() < 0.5

    def test_gate_activation(self):
        """Gate should activate for matching queries."""
        # Store a correction
        key = F.normalize(torch.randn(self.dim), dim=0)
        value = torch.randn(self.dim) * 0.1
        self.memory.write(key, value, word="test")

        # Query with the exact same key (should activate)
        embeddings = key.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]

        # Lower τ to make activation easier for testing
        self.refiner.tau.data = torch.tensor(0.0)

        refined, gate = self.refiner(embeddings)

        # Gate should be active for the matching query
        assert gate.max().item() > 0.5

    def test_2d_input(self):
        """Refiner should handle 2D input [seq_len, dim]."""
        self.memory.write(
            F.normalize(torch.randn(self.dim), dim=0),
            torch.randn(self.dim),
            word="test",
        )

        embeddings = torch.randn(5, self.dim)
        refined, gate = self.refiner(embeddings)

        assert refined.shape == (5, self.dim)
        assert gate.shape == (5,)

    def test_gate_analysis(self):
        """Gate analysis should return per-token info."""
        self.memory.write(
            F.normalize(torch.randn(self.dim), dim=0),
            torch.randn(self.dim),
            word="test_word",
        )

        embeddings = torch.randn(5, self.dim)
        analysis = self.refiner.get_gate_analysis(
            embeddings,
            token_texts=["The", "quick", "brown", "fox", "jumps"],
        )

        assert len(analysis) == 5
        assert "token_idx" in analysis[0]
        assert "gate_value" in analysis[0]
        assert "token_text" in analysis[0]
        assert analysis[0]["token_text"] == "The"


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_per_identical(self):
        from flowedit.utils.metrics import compute_per
        per = compute_per(["a", "b", "c"], ["a", "b", "c"])
        assert per == 0.0

    def test_per_different(self):
        from flowedit.utils.metrics import compute_per
        per = compute_per(["x", "y", "z"], ["a", "b", "c"])
        assert per == 1.0  # All substitutions

    def test_per_empty_reference(self):
        from flowedit.utils.metrics import compute_per
        per = compute_per([], [])
        assert per == 0.0

    def test_mel_loss_zero(self):
        from flowedit.utils.metrics import compute_mel_loss
        mel = torch.randn(1, 80, 100)
        loss = compute_mel_loss(mel, mel)
        assert loss.item() < 1e-6

    def test_mel_loss_positive(self):
        from flowedit.utils.metrics import compute_mel_loss
        mel1 = torch.randn(1, 80, 100)
        mel2 = torch.randn(1, 80, 100)
        loss = compute_mel_loss(mel1, mel2)
        assert loss.item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
