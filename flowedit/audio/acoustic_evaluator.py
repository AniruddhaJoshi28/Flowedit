"""
Acoustic Feasibility & Latent Intervention Evaluator for F5-TTS.

Evaluates whether controlled latent interventions in frozen F5-TTS can reliably
alter target pronunciation while strictly preserving non-target speech.

Zero hardcoding: operates on arbitrary carrier sentences, target spans, and audio.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_spectral_metrics(
    mel_baseline: torch.Tensor,
    mel_intervened: torch.Tensor,
    target_frame_span: Tuple[int, int],
) -> Dict[str, float]:
    """Computes target acoustic shift and non-target speech drift.
    
    Args:
        mel_baseline: [1, n_mels, n_frames] or [n_mels, n_frames]
        mel_intervened: [1, n_mels, n_frames] or [n_mels, n_frames]
        target_frame_span: (start_frame, end_frame) of target word
    Returns:
        Dict with target_mse, target_spectral_convergence, non_target_mse, drift_ratio
    """
    b_mel = mel_baseline.squeeze(0).cpu().float()
    i_mel = mel_intervened.squeeze(0).cpu().float()
    
    # Align frame counts if slight length mismatch
    min_len = min(b_mel.shape[-1], i_mel.shape[-1])
    b_mel = b_mel[:, :min_len]
    i_mel = i_mel[:, :min_len]
    
    f_start, f_end = target_frame_span
    f_start = max(0, min(f_start, min_len - 1))
    f_end = max(f_start + 1, min(f_end, min_len))
    
    # 1. Target Region
    target_b = b_mel[:, f_start:f_end]
    target_i = i_mel[:, f_start:f_end]
    
    target_diff = target_i - target_b
    target_mse = float(torch.mean(target_diff ** 2).item())
    
    norm_base = float(torch.norm(target_b, p="fro").item()) + 1e-8
    target_spec_conv = float(torch.norm(target_diff, p="fro").item()) / norm_base
    
    # 2. Non-Target Region (Prefix + Suffix frames)
    non_target_masks = torch.ones(min_len, dtype=torch.bool)
    non_target_masks[f_start:f_end] = False
    
    if non_target_masks.any():
        non_target_b = b_mel[:, non_target_masks]
        non_target_i = i_mel[:, non_target_masks]
        non_target_diff = non_target_i - non_target_b
        non_target_mse = float(torch.mean(non_target_diff ** 2).item())
        non_target_norm = float(torch.norm(non_target_b, p="fro").item()) + 1e-8
        non_target_spec_conv = float(torch.norm(non_target_diff, p="fro").item()) / non_target_norm
    else:
        non_target_mse = 0.0
        non_target_spec_conv = 0.0
        
    # Drift ratio: how much non-target changed relative to target change
    drift_ratio = float(non_target_mse / (target_mse + 1e-8))
    
    # Anomaly / energy explosion check
    max_energy_ratio = float(torch.max(i_mel.abs()).item()) / (float(torch.max(b_mel.abs()).item()) + 1e-8)
    
    return {
        "target_mse": round(target_mse, 5),
        "target_spectral_convergence": round(target_spec_conv, 4),
        "non_target_mse": round(non_target_mse, 5),
        "non_target_spectral_convergence": round(non_target_spec_conv, 4),
        "drift_ratio": round(drift_ratio, 4),
        "max_energy_ratio": round(max_energy_ratio, 3),
        "is_acoustically_localized": bool(drift_ratio < 0.15 and target_mse > 0.005),
    }
