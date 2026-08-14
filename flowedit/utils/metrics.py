"""
Evaluation metrics for FlowEdit.

Implements Phoneme Error Rate (PER) and Mel-Cepstral Distortion (MCD)
as used in the paper (Section 4.1).
"""

import torch
import numpy as np
from typing import List, Optional, Tuple


def compute_per(
    predicted_phonemes: List[str],
    reference_phonemes: List[str],
) -> float:
    """Compute Phoneme Error Rate using edit distance.

    PER = (substitutions + insertions + deletions) / len(reference)

    Paper uses wav2vec 2.0-based phoneme recognizer for automatic PER.
    This function computes PER given two phoneme sequences.

    Args:
        predicted_phonemes: List of predicted phoneme strings
        reference_phonemes: List of reference phoneme strings

    Returns:
        PER as a float in [0, 1] (multiply by 100 for percentage)
    """
    if len(reference_phonemes) == 0:
        return 0.0 if len(predicted_phonemes) == 0 else 1.0

    # Levenshtein distance via dynamic programming
    n = len(predicted_phonemes)
    m = len(reference_phonemes)

    # dp[i][j] = edit distance between pred[:i] and ref[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if predicted_phonemes[i - 1] == reference_phonemes[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    edit_distance = dp[n][m]
    per = edit_distance / m

    return per


def compute_mcd(
    mel_pred: torch.Tensor,
    mel_ref: torch.Tensor,
    n_mfcc: int = 13,
) -> float:
    """Compute Mel-Cepstral Distortion between two mel-spectrograms.

    MCD measures spectral distance and is commonly used for TTS evaluation.
    Lower MCD = better spectral match.

    Paper Table 1: FlowEdit achieves MCD of 3.22 vs 6.82 baseline.

    MCD = (10√2 / ln10) × mean(||mfcc_pred - mfcc_ref||₂)

    Args:
        mel_pred: Predicted mel-spectrogram [n_mels, T]
        mel_ref: Reference mel-spectrogram [n_mels, T]
        n_mfcc: Number of MFCCs to use (typically 13, excluding c0)

    Returns:
        MCD value in dB (lower is better)
    """
    # Align lengths
    min_len = min(mel_pred.shape[-1], mel_ref.shape[-1])
    mel_pred = mel_pred[..., :min_len]
    mel_ref = mel_ref[..., :min_len]

    # Move to numpy for DCT computation
    mel_pred_np = mel_pred.detach().cpu().numpy()
    mel_ref_np = mel_ref.detach().cpu().numpy()

    # Handle batched input
    if mel_pred_np.ndim == 3:
        mel_pred_np = mel_pred_np[0]
        mel_ref_np = mel_ref_np[0]

    # Compute MFCCs via DCT of mel (skip c0 = energy)
    from scipy.fft import dct

    mfcc_pred = dct(mel_pred_np, type=2, axis=0, norm="ortho")[1: n_mfcc + 1]
    mfcc_ref = dct(mel_ref_np, type=2, axis=0, norm="ortho")[1: n_mfcc + 1]

    # Frame-wise Euclidean distance
    diff = mfcc_pred - mfcc_ref
    frame_dist = np.sqrt(np.sum(diff ** 2, axis=0))

    # MCD scaling factor: 10√2 / ln(10) ≈ 6.1415
    scale = 10.0 * np.sqrt(2.0) / np.log(10.0)

    mcd = scale * np.mean(frame_dist)

    return float(mcd)


def compute_mel_loss(
    mel_pred: torch.Tensor,
    mel_ref: torch.Tensor,
) -> torch.Tensor:
    """Compute the mel-spectrogram reconstruction loss.

    Normalized per-frame energy removes overall volume/gain differences,
    focusing gradient optimization purely on phonetic formant spectral match.
    Combined L1 + MSE loss provides sharp spectral resolution.

    Args:
        mel_pred: Predicted mel [batch, n_mels, T]
        mel_ref: Reference mel [batch, n_mels, T]

    Returns:
        Scalar loss tensor (differentiable)
    """
    # Ensure 3D shape [batch, n_mels, T]
    if mel_pred.dim() == 2:
        mel_pred = mel_pred.unsqueeze(0)
    if mel_ref.dim() == 2:
        mel_ref = mel_ref.unsqueeze(0)

    # Align time dimensions via linear interpolation
    if mel_pred.shape[-1] != mel_ref.shape[-1]:
        mel_pred = torch.nn.functional.interpolate(
            mel_pred,
            size=mel_ref.shape[-1],
            mode="linear",
            align_corners=False,
        )

    # Mean-normalize time dimension to eliminate microphone/recording gain differences
    mel_pred_norm = mel_pred - mel_pred.mean(dim=-1, keepdim=True)
    mel_ref_norm = mel_ref - mel_ref.mean(dim=-1, keepdim=True)

    # Combined L1 (formant resolution) + L2 (overall spectral shape) loss
    l1_loss = torch.nn.functional.l1_loss(mel_pred_norm, mel_ref_norm)
    l2_loss = torch.nn.functional.mse_loss(mel_pred_norm, mel_ref_norm)

    loss = l1_loss + l2_loss
    return loss

def compute_f0_loss(
    audio_pred: torch.Tensor,
    audio_ref: torch.Tensor,
    sample_rate: int = 22050,
) -> torch.Tensor:
    """Compute F0 (pitch) RMSE loss for tonal language support.

    Paper Section 4.5: Adding α·L_F0 to the mel loss reduces
    Mandarin PER from 9.1% → 6.4% and Vietnamese from 5.3% → 4.1%.

    Uses CREPE as specified in the paper to extract accurate pitch tracks.

    Args:
        audio_pred: Predicted audio waveform [1, T]
        audio_ref: Reference audio waveform [1, T]
        sample_rate: Sample rate

    Returns:
        F0 RMSE loss (scalar tensor)
    """
    # Align to shorter length
    min_len = min(audio_pred.shape[-1], audio_ref.shape[-1])
    audio_pred = audio_pred[..., :min_len]
    audio_ref = audio_ref[..., :min_len]

    try:
        import torchcrepe
        # CREPE expects 16kHz audio, resample if necessary
        if sample_rate != 16000:
            import torchaudio.functional as F_audio
            audio_pred_16k = F_audio.resample(audio_pred, sample_rate, 16000)
            audio_ref_16k = F_audio.resample(audio_ref, sample_rate, 16000)
        else:
            audio_pred_16k = audio_pred
            audio_ref_16k = audio_ref

        # Compute pitch tracks using CREPE
        pitch_pred, _ = torchcrepe.predict(
            audio_pred_16k, 16000, 160, 50, 2000, "full", return_periodicity=True
        )
        pitch_ref, _ = torchcrepe.predict(
            audio_ref_16k, 16000, 160, 50, 2000, "full", return_periodicity=True
        )

        min_frames = min(pitch_pred.shape[-1], pitch_ref.shape[-1])
        pitch_pred = pitch_pred[..., :min_frames]
        pitch_ref = pitch_ref[..., :min_frames]

        # RMSE loss
        f0_loss = torch.sqrt(torch.mean((pitch_pred - pitch_ref) ** 2) + 1e-8)
        
    except ImportError:
        # Differentiable GPU-accelerated pitch/spectral centroid tracking fallback
        device = audio_pred.device
        stft_pred = torch.stft(audio_pred, n_fft=512, hop_length=160, win_length=512, return_complex=True)
        stft_ref = torch.stft(audio_ref, n_fft=512, hop_length=160, win_length=512, return_complex=True)
        
        mag_pred = torch.abs(stft_pred)
        mag_ref = torch.abs(stft_ref)
        
        min_f = min(mag_pred.shape[-1], mag_ref.shape[-1])
        mag_pred = mag_pred[..., :min_f]
        mag_ref = mag_ref[..., :min_f]
        
        freqs = torch.linspace(0, sample_rate // 2, steps=mag_pred.shape[1], device=device).view(1, -1, 1)
        centroid_pred = torch.sum(freqs * mag_pred, dim=1) / (torch.sum(mag_pred, dim=1) + 1e-6)
        centroid_ref = torch.sum(freqs * mag_ref, dim=1) / (torch.sum(mag_ref, dim=1) + 1e-6)
        
        f0_loss = torch.sqrt(torch.mean((centroid_pred - centroid_ref) ** 2) + 1e-8)

    return f0_loss
