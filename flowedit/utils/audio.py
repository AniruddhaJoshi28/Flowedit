"""
Audio processing utilities for FlowEdit.

Provides mel-spectrogram extraction, audio I/O, and data augmentation
functions used throughout the pipeline.
"""

import torch
try:
    import torchaudio
except (ImportError, OSError):
    torchaudio = None
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Tuple
from pathlib import Path

from flowedit.config import AudioConfig


class AudioProcessor:
    """Central audio processing class for FlowEdit.

    Handles mel-spectrogram computation, audio loading/saving,
    and data augmentation for the optimization loop.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self._mel_transform = None

    @property
    def mel_transform(self) -> torchaudio.transforms.MelSpectrogram:
        """Lazily initialize mel-spectrogram transform."""
        if self._mel_transform is None:
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.config.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
                n_mels=self.config.n_mels,
                f_min=self.config.fmin,
                f_max=self.config.fmax,
                power=2.0,
                normalized=False,
            )
        return self._mel_transform

    def load_audio(
        self,
        path: str,
        target_sr: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Load audio file and resample if necessary.

        Args:
            path: Path to audio file (wav, mp3, flac, etc.)
            target_sr: Target sample rate. Defaults to config sample_rate.

        Returns:
            Tuple of (waveform tensor [1, T], sample_rate)

        Raises:
            FileNotFoundError: If audio file doesn't exist.
            ValueError: If audio is too short or too long.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        target_sr = target_sr or self.config.sample_rate

        import librosa
        import numpy as np
        
        # librosa automatically converts to mono and resamples to target_sr safely without FFmpeg/torchcodec
        y, sr = librosa.load(str(path), sr=target_sr, mono=True)
        
        # Convert numpy array back to PyTorch tensor format (1, T)
        waveform = torch.from_numpy(y).unsqueeze(0)

        # Validate duration
        duration = waveform.shape[1] / sr
        if duration < self.config.ref_audio_min_duration:
            raise ValueError(
                f"Audio too short: {duration:.1f}s "
                f"(minimum: {self.config.ref_audio_min_duration}s). "
                f"Paper recommends ≥1.5s for optimal results."
            )

        if duration > self.config.ref_audio_max_duration:
            # Trim to max duration (quality plateaus beyond 3s per paper)
            max_samples = int(self.config.ref_audio_max_duration * sr)
            waveform = waveform[:, :max_samples]

        return waveform, sr

    def compute_mel(
        self,
        waveform: torch.Tensor,
        normalize: bool = True,
        n_mels: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute mel-spectrogram from waveform.

        Args:
            waveform: Audio tensor [batch, T] or [1, T]
            normalize: If True, apply log scaling and normalization.
            n_mels: Optional override for number of mel channels.

        Returns:
            Mel-spectrogram tensor [batch, n_mels, time_frames]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        n_mels_val = n_mels or self.config.n_mels
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            n_mels=n_mels_val,
            f_min=self.config.fmin,
            f_max=self.config.fmax,
            power=2.0,
            normalized=False,
        ).to(waveform.device)

        mel = mel_transform(waveform)

        if normalize:
            # Log-mel spectrogram (standard for TTS loss computation)
            mel = torch.log(mel + 1e-5)

        return mel

    def save_audio(
        self,
        waveform: torch.Tensor,
        path: str,
        sample_rate: Optional[int] = None,
    ) -> None:
        """Save audio tensor to file.

        Args:
            waveform: Audio tensor [1, T] or [T]
            path: Output file path
            sample_rate: Sample rate (defaults to config)
        """
        sr = sample_rate or self.config.sample_rate
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Move to CPU for saving
        waveform = waveform.detach().cpu()

        torchaudio.save(str(path), waveform, sr)

    def augment_mel(
        self,
        mel: torch.Tensor,
        time_stretch_range: Tuple[float, float] = (0.9, 1.1),
        gain_db_range: Tuple[float, float] = (-3.0, 3.0),
    ) -> torch.Tensor:
        """Apply data augmentation to mel-spectrogram.

        Paper Section 3.2: "Data augmentation (time-stretching and gain
        scaling in mel space) is applied to the reference to promote
        robust latents."

        Args:
            mel: Mel-spectrogram [batch, n_mels, T]
            time_stretch_range: Range for time stretch factor
            gain_db_range: Range for gain scaling in dB

        Returns:
            Augmented mel-spectrogram (same shape as input or slightly
            different in time dimension due to stretching)
        """
        # Random gain scaling in log-mel space (equivalent to dB shift)
        gain_db = torch.empty(1).uniform_(*gain_db_range).item()
        gain_linear = 10 ** (gain_db / 20.0)
        mel_aug = mel + np.log(gain_linear)

        # Time stretching via interpolation in mel space
        stretch_factor = torch.empty(1).uniform_(*time_stretch_range).item()
        if abs(stretch_factor - 1.0) > 0.01:
            # Interpolate time axis
            new_len = int(mel.shape[-1] * stretch_factor)
            mel_aug = torch.nn.functional.interpolate(
                mel_aug,
                size=new_len,
                mode="linear",
                align_corners=False,
            )

        return mel_aug

    def align_mel_lengths(
        self,
        mel_pred: torch.Tensor,
        mel_ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align two mel-spectrograms to the same time length.

        Pads the shorter one with the minimum value (silence in log-mel).

        Args:
            mel_pred: Predicted mel [batch, n_mels, T1]
            mel_ref: Reference mel [batch, n_mels, T2]

        Returns:
            Tuple of aligned mel spectrograms (both [batch, n_mels, max(T1,T2)])
        """
        len_pred = mel_pred.shape[-1]
        len_ref = mel_ref.shape[-1]

        if len_pred == len_ref:
            return mel_pred, mel_ref

        target_len = min(len_pred, len_ref)

        # Truncate the longer one
        mel_pred = mel_pred[..., :target_len]
        mel_ref = mel_ref[..., :target_len]

        return mel_pred, mel_ref

    def extract_segment(
        self,
        waveform: torch.Tensor,
        start_time: float,
        end_time: float,
        sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract a time segment from a waveform.

        Args:
            waveform: Audio tensor [1, T]
            start_time: Segment start in seconds
            end_time: Segment end in seconds
            sample_rate: Sample rate (defaults to config)

        Returns:
            Segment tensor [1, segment_length]
        """
        sr = sample_rate or self.config.sample_rate
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        start_sample = max(0, start_sample)
        end_sample = min(waveform.shape[-1], end_sample)

        return waveform[..., start_sample:end_sample]
