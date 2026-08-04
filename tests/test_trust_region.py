"""
Unit tests for trust-region projection function project_delta.
"""

import pytest
import torch
from flowedit.optimizer.latent_optimizer import project_delta, TRUST_REGION_RATIO


def test_project_delta_within_ratio():
    base = torch.randn(1, 10, 512)
    base_norm = base.norm(dim=-1, keepdim=True)
    
    # Small delta -> should not be scaled down
    delta = 0.01 * torch.randn(1, 10, 512)
    projected = project_delta(delta, base, ratio=0.03)

    assert torch.allclose(projected, delta)


def test_project_delta_exceeds_ratio():
    base = torch.ones(1, 10, 512)
    # Base L2 norm along dim=-1 is sqrt(512) ≈ 22.627
    
    # Large delta with norm = 100.0
    delta = torch.ones(1, 10, 512) * (100.0 / (512 ** 0.5))
    
    # Project with ratio = 0.03
    projected = project_delta(delta, base, ratio=0.03)
    
    proj_norm = projected.norm(dim=-1, keepdim=True)
    max_allowed_norm = 0.03 * base.norm(dim=-1, keepdim=True)

    # Projected norm should be clamped to max_allowed_norm
    assert torch.allclose(proj_norm, max_allowed_norm, atol=1e-4)


def test_trust_region_ratios_config():
    assert "f5_conditioning" in TRUST_REGION_RATIO
    assert "xtts_gpt_conditioning" in TRUST_REGION_RATIO
    assert "cosyvoice_adapter_hidden" in TRUST_REGION_RATIO
    assert TRUST_REGION_RATIO["f5_conditioning"] == 0.03
