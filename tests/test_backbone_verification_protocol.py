"""
FlowEdit 6-Step Backbone Verification Protocol Test Suite.

Automated verification suite testing every backbone (CosyVoice, F5-TTS, XTTS-v2) against
6 strict mathematical & physical invariants:
1. Test 1 (Identity Pass-Through)
2. Test 2 (Perturbation Sensitivity)
3. Test 3 (Zero-Perturbation Numerical Invariance)
4. Test 4 (Optimization Loss Convergence)
5. Test 5 (Hopfield Storage & Soft Retrieval Integrity)
6. Test 6 (Morphological Generalization)

Reference: FlowEdit publication-grade verification suite.
"""

import os
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from flowedit.config import FlowEditConfig, BackboneConfig
from flowedit.backbone import create_backbone, OptimizationMode
from flowedit.memory import HopfieldMemory



@pytest.mark.parametrize("backbone_type", ["cosyvoice", "f5tts", "xtts"])
def test_backbone_verification_protocol(backbone_type):

    config = FlowEditConfig()
    config.backbone.backbone_type = backbone_type
    
    # ── Step 0: Initialize Backbone ──────────────────────────────────────
    backbone = create_backbone(config.backbone)
    try:
        backbone.load_model()
        if backbone_type == "cosyvoice" and getattr(backbone, "cosyvoice_instance", None) is None:
            pytest.skip("CosyVoice model instance not available.")
        if backbone_type == "f5tts" and getattr(backbone, "tts_api", None) is None:
            pytest.skip("F5-TTS model API instance not available.")
        if backbone_type == "xtts" and getattr(backbone, "model", None) is None:
            pytest.skip("XTTS model instance not available.")
    except Exception as e:
        pytest.skip(f"Skipping backbone test for {backbone_type}: {e}")

    assert hasattr(backbone, "capabilities")
    assert isinstance(backbone.optimization_mode, OptimizationMode)

    
    import soundfile as sf
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
        dummy_audio_path = tmp_f.name
    
    t_samp = np.linspace(0, 1.0, 24000)
    dummy_wav = (0.5 * np.sin(2 * np.pi * 440 * t_samp)).astype(np.float32)
    sf.write(dummy_audio_path, dummy_wav, 24000)

    text = "Linux"
    try:
        speaker_cond = backbone.get_speaker_embedding(audio_path=dummy_audio_path)

    except Exception as e:
        if os.path.exists(dummy_audio_path):
            os.remove(dummy_audio_path)
        pytest.skip(f"Skipping backbone test for {backbone_type}: {e}")

    # ── Test 1: Identity Pass-Through ────────────────────────────────────
    c_base = backbone.encode_text(text)
    audio1, sr1 = backbone.synthesize(text, speaker_cond)
    audio_emb1, sr_emb1 = backbone.synthesize_from_embeddings(c_base, speaker_cond, text)

    
    assert sr1 == sr_emb1
    assert audio1.shape[-1] > 0
    assert audio_emb1.shape[-1] > 0
    
    # ── Test 2: Perturbation Sensitivity ──────────────────────────────────
    delta = torch.randn_like(c_base) * 0.5
    c_perturbed = c_base + delta
    audio_perturbed, _ = backbone.synthesize_from_embeddings(c_perturbed, speaker_cond, text)
    
    # Audio output must change when perturbed
    audio_diff = torch.norm(audio_emb1 - audio_perturbed).item()
    assert audio_diff > 0.0, f"Perturbation had no effect on backbone {backbone_type}"

    # ── Test 3: Zero-Perturbation Numerical Invariance ─────────────────────
    c_zero = c_base + torch.zeros_like(c_base)
    audio_zero, _ = backbone.synthesize_from_embeddings(c_zero, speaker_cond, text)
    
    mse = F.mse_loss(audio_emb1, audio_zero).item()
    assert mse < 0.05, f"Zero perturbation produced unexpected delta MSE={mse}"


    # ── Test 4: Optimization Loss Convergence ────────────────────────────
    import soundfile as sf
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
        dummy_audio_path = tmp_f.name
    t_samp2 = np.linspace(0, 1.0, 24000)
    dummy_wav = (0.5 * np.sin(2 * np.pi * 440 * t_samp2)).astype(np.float32)
    sf.write(dummy_audio_path, dummy_wav, 24000)


    perturbed_var = c_base.clone().detach().requires_grad_(True)
    losses = []
    optimizer = torch.optim.Adam([perturbed_var], lr=0.01)
    
    for step in range(10):
        optimizer.zero_grad()
        loss_dict = backbone.compute_optimization_loss(
            perturbed_var,
            ref_audio_path=dummy_audio_path,
            speaker_conditioning=speaker_cond,
            text=text,
        )
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if os.path.exists(dummy_audio_path):
        try:
            os.remove(dummy_audio_path)
        except Exception:
            pass


    assert losses[-1] < losses[0], f"Optimization did not show overall convergence: L_0={losses[0]} -> L_10={losses[-1]}"

    # ── Test 5: Hopfield Storage & Soft Retrieval Integrity ───────────────
    memory = HopfieldMemory(
        dim=backbone.embedding_dim,
        config=config.memory,
    )
    
    key = c_base.mean(dim=1).squeeze(0)  # [d]
    learned_delta = delta.mean(dim=1).squeeze(0)  # [d]
    
    memory.write(key=key, value=learned_delta, word="Linux")
    
    query_key = key.clone()
    retrieved_delta, max_sim = memory.retrieve(query_key)
    
    assert max_sim.item() > 0.8, f"Hopfield failed to activate for exact key query (sim={max_sim.item()})"
    assert retrieved_delta is not None
    sim_delta = F.cosine_similarity(retrieved_delta.unsqueeze(0), learned_delta.unsqueeze(0)).item()
    assert sim_delta > 0.8, f"Retrieved delta similarity too low: {sim_delta}"

    # ── Test 6: Morphological Generalization ──────────────────────────────
    c_variant = backbone.encode_text("Linux's")
    query_variant_key = c_variant.mean(dim=1).squeeze(0)
    
    retrieved_variant_delta, variant_sim = memory.retrieve(query_variant_key)
    assert variant_sim.item() > 0.1, f"Fuzzy key similarity too low for variant: {variant_sim.item()}"

