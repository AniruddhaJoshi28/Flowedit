import torch

from flowedit.config import MemoryConfig
from flowedit.memory.hopfield_memory import HopfieldMemory
from flowedit.refiner.hopfield_refiner import HopfieldRefiner


def make_memory(dim=16):
    config = MemoryConfig(gate_threshold_init=0.4, perturbation_scale=1.0)
    return HopfieldMemory(dim, config), config


def test_exact_word_uses_own_entry_even_when_other_key_is_identical():
    memory, config = make_memory()
    text = "Say Nguyen and Siobhan"
    embeddings = torch.randn(1, len(text), 16)
    key = embeddings[0, 4:10].mean(0)
    memory.write(key, torch.ones(6, 16), word="Nguyen")
    memory.write(key, torch.full((7, 16), 9.0), word="Siobhan")

    refined, gates = HopfieldRefiner(memory, config)(embeddings, text=text)

    assert torch.allclose(refined[0, 4:10] - embeddings[0, 4:10], torch.ones(6, 16))
    assert (gates[0, 4:10] == 1).all()


def test_all_case_insensitive_occurrences_are_corrected():
    memory, config = make_memory()
    text = "Nguyen met NGUYEN."
    embeddings = torch.randn(1, len(text), 16)
    memory.write(torch.randn(16), torch.full((6, 16), 2.0), word="Nguyen")

    refined, gates = HopfieldRefiner(memory, config)(embeddings, text=text)

    for start in (0, 11):
        assert (gates[0, start:start + 6] == 1).all()
        assert torch.allclose(
            refined[0, start:start + 6] - embeddings[0, start:start + 6],
            torch.full((6, 16), 2.0),
        )


def test_substrings_and_unmentioned_words_are_never_modified():
    memory, config = make_memory()
    text = "A program runs."
    embeddings = torch.randn(1, len(text), 16)
    memory.write(torch.randn(16), torch.ones(3, 16), word="ram")

    refined, gates = HopfieldRefiner(memory, config)(embeddings, text=text)

    assert torch.equal(refined, embeddings)
    assert (gates == 0).all()


def test_deduplication_never_merges_different_spellings():
    memory, _ = make_memory()
    key = torch.randn(16)
    memory.write(key, torch.ones(16), word="Indian")
    memory.write(key, torch.zeros(16), word="Japanese")
    assert memory.size == 2


def test_unicode_english_name_is_matched():
    memory, config = make_memory()
    text = "Zo? speaks English."
    embeddings = torch.randn(1, len(text), 16)
    memory.write(torch.randn(16), torch.ones(3, 16), word="zo?")

    refined, gates = HopfieldRefiner(memory, config)(embeddings, text=text)

    assert (gates[0, 0:3] == 1).all()
    assert not torch.equal(refined[0, 0:3], embeddings[0, 0:3])
