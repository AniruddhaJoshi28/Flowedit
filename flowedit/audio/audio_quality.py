"""
Audio Quality & Artifact Assessment Metrics for FlowEdit.

Provides calculations for SNR, RMS, spectral flatness, spectral entropy,
silence ratio, and clipping metrics used in candidate selection and validation.
"""

import torch
import numpy as np
from typing import Dict, Any


def compute_rms(waveform: torch.Tensor) -> float:
    """Computes Root Mean Square (RMS) energy."""
    return torch.sqrt(torch.mean(waveform ** 2)).item()


def compute_peak(waveform: torch.Tensor) -> float:
    """Computes peak amplitude."""
    return torch.max(torch.abs(waveform)).item()


def compute_clipping_ratio(waveform: torch.Tensor, threshold: float = 0.98) -> float:
    """Computes ratio of samples exceeding peak threshold."""
    abs_wf = torch.abs(waveform)
    clipping_samples = torch.sum(abs_wf >= threshold).item()
    return clipping_samples / max(1, waveform.numel())


def compute_silence_ratio(waveform: torch.Tensor, threshold_db: float = -40.0) -> float:
    """Computes ratio of silent samples below threshold dB."""
    peak = torch.max(torch.abs(waveform)).item()
    thresh = max(1e-5, peak * (10 ** (threshold_db / 20.0)))
    silent_samples = torch.sum(torch.abs(waveform) < thresh).item()
    return silent_samples / max(1, waveform.numel())


def compute_spectral_flatness(waveform: torch.Tensor, n_fft: int = 1024) -> float:
    """Computes spectral flatness (ratio of geometric to arithmetic mean of power spectrum)."""
    if waveform.ndim > 1:
        wf = waveform.squeeze()
    else:
        wf = waveform
        
    wf_np = wf.cpu().numpy()
    if len(wf_np) < n_fft:
        return 0.5

    spec = np.abs(np.fft.rfft(wf_np, n=n_fft)) ** 2
    spec = spec + 1e-10  # Prevent log(0)

    geometric_mean = np.exp(np.mean(np.log(spec)))
    arithmetic_mean = np.mean(spec)

    return float(geometric_mean / arithmetic_mean)


def compute_audio_quality_metrics(waveform: torch.Tensor, sample_rate: int) -> Dict[str, float]:
    """Computes comprehensive audio quality metrics for output validation."""
    peak = compute_peak(waveform)
    rms = compute_rms(waveform)
    clipping = compute_clipping_ratio(waveform)
    silence = compute_silence_ratio(waveform)
    flatness = compute_spectral_flatness(waveform)
    duration = waveform.shape[-1] / float(sample_rate)

    return {
        "peak": peak,
        "rms": rms,
        "clipping_ratio": clipping,
        "silence_ratio": silence,
        "spectral_flatness": flatness,
        "duration_seconds": duration,
    }
