"""
Shared Internal Audio Preprocessor for FlowEdit.

All reference audio must pass through one deterministic preprocessing path.
Provides PreparedAudio dataclass and standard audio normalization routines.
"""

from dataclasses import dataclass
import hashlib
import numpy as np
import torch
try:
    import torchaudio
    import torchaudio.transforms as T
except (ImportError, OSError, Exception):
    torchaudio = None
    T = None
import logging

logger = logging.getLogger(__name__)


@dataclass
class PreparedAudio:
    waveform: torch.Tensor          # Contiguous float32 tensor [1, T]
    sample_rate: int                # Backbone target sample rate
    duration_seconds: float        # Audio duration in seconds
    source_path: str               # Original file path
    checksum: str                  # SHA256 checksum of raw audio file
    peak: float                    # Peak amplitude (0.0 to 1.0)
    rms: float                     # Root mean square energy
    clipping_ratio: float          # Fraction of samples at or near peak (>= 0.99)
    silence_ratio: float           # Fraction of samples below silence threshold


class AudioPreprocessor:
    """Deterministic audio preprocessor for reference speaker audio."""

    def __init__(
        self,
        target_sample_rate: int = 24000,
        trailing_silence_sec: float = 0.45,
        silence_threshold_db: float = -45.0,
        peak_norm_target: float = 0.95,
    ):
        self.target_sample_rate = target_sample_rate
        self.trailing_silence_sec = trailing_silence_sec
        self.silence_threshold_db = silence_threshold_db
        self.peak_norm_target = peak_norm_target

    def preprocess(self, file_path: str, target_sr: int = None) -> PreparedAudio:
        """Processes reference audio file through the 14-step deterministic pipeline.

        1. Decode file & check validity.
        2. Convert integer PCM to float32 [-1.0, 1.0].
        3. Convert stereo/multichannel to mono via channel averaging.
        4. Remove NaN and infinite values.
        5. Remove DC offset.
        6. Resample to target sample rate.
        7. Apply conservative peak normalization.
        8. Trim excessive leading and trailing silence.
        9. Retain ~300-600ms trailing silence.
        10. Calculate metrics (checksum, peak, RMS, clipping ratio, silence ratio).
        11. Return contiguous tensor PreparedAudio.
        """
        sr_out = target_sr or self.target_sample_rate

        # Read raw checksum
        with open(file_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        # Step 1: Decode file with soundfile / torchaudio fallbacks
        waveform = None
        sr = None

        try:
            if torchaudio is not None:
                waveform, sr = torchaudio.load(file_path)
            else:
                raise ImportError("torchaudio not installed")
        except Exception as e_ta:
            logger.debug(f"torchaudio.load failed ({e_ta}), attempting soundfile fallback...")
            try:
                import soundfile as sf
                data, sr = sf.read(file_path)
                data_tensor = torch.from_numpy(data).float()
                if data_tensor.ndim == 1:
                    waveform = data_tensor.unsqueeze(0)
                else:
                    waveform = data_tensor.T
            except Exception as e_sf:
                try:
                    import wave
                    with wave.open(file_path, "rb") as wf:
                        sr = wf.getframerate()
                        n_channels = wf.getnchannels()
                        n_frames = wf.getnframes()
                        frames = wf.readframes(n_frames)
                        dtype = np.int16 if wf.getsampwidth() == 2 else np.int32
                        audio_np = np.frombuffer(frames, dtype=dtype).astype(np.float32) / 32768.0
                        if n_channels > 1:
                            audio_np = audio_np.reshape(-1, n_channels).T
                        else:
                            audio_np = audio_np.reshape(1, -1)
                        waveform = torch.from_numpy(audio_np)
                except Exception as e_wave:
                    raise ValueError(f"Failed to decode audio file '{file_path}': {e_ta} | {e_sf} | {e_wave}") from e_ta

        if waveform is None or waveform.numel() == 0:
            raise ValueError(f"Audio file '{file_path}' contains zero samples.")


        # Step 2: Convert to float32
        waveform = waveform.to(torch.float32)

        # Step 3: Mono conversion via channel averaging
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Step 4: Remove NaN and Infinite values
        waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 5: Remove DC offset
        waveform = waveform - torch.mean(waveform)

        # Step 6: Resample if necessary
        if sr != sr_out:
            resampled = False
            if T is not None and hasattr(T, "Resample"):
                try:
                    resampler = T.Resample(sr, sr_out)
                    waveform = resampler(waveform)
                    resampled = True
                except Exception as e_resamp:
                    logger.debug(f"torchaudio T.Resample failed ({e_resamp}), falling back...")
            if not resampled:
                try:
                    import scipy.signal
                    resampled_np = scipy.signal.resample(waveform.squeeze().numpy(), int(waveform.shape[-1] * sr_out / sr))
                    waveform = torch.from_numpy(resampled_np).unsqueeze(0).float()
                    resampled = True
                except Exception:
                    pass
            if not resampled:
                # Robust pure-PyTorch 1D linear interpolation fallback (zero extra dependencies)
                try:
                    target_length = max(1, int(round(waveform.shape[-1] * (float(sr_out) / float(sr)))))
                    # waveform is [1, T] -> interpolate expects [N, C, L]
                    w_3d = waveform.unsqueeze(0)
                    w_resampled = torch.nn.functional.interpolate(
                        w_3d,
                        size=target_length,
                        mode="linear",
                        align_corners=False
                    )
                    waveform = w_resampled.squeeze(0)
                    resampled = True
                    logger.debug(f"Resampled audio from {sr}Hz to {sr_out}Hz using torch linear interpolation.")
                except Exception as e_interp:
                    logger.warning(f"Could not resample audio from {sr}Hz to {sr_out}Hz: {e_interp}")

        # Step 7: Conservative peak normalization
        peak_val = torch.max(torch.abs(waveform)).item()
        if peak_val > 1e-6:
            waveform = waveform * (self.peak_norm_target / peak_val)

        # Step 8 & 9: Trim silence & retain natural trailing silence
        waveform = self._trim_silence_with_padding(waveform, sr_out)

        # Step 10: Metric calculations
        abs_wf = torch.abs(waveform)
        peak = torch.max(abs_wf).item()
        rms = torch.sqrt(torch.mean(waveform ** 2)).item()

        # Clipping ratio: samples near peak threshold (>= 0.98)
        clipping_samples = torch.sum(abs_wf >= 0.98).item()
        clipping_ratio = clipping_samples / max(1, waveform.shape[-1])

        # Silence ratio: samples below threshold (-45 dB relative to peak)
        silence_thresh = max(1e-5, peak * (10 ** (self.silence_threshold_db / 20.0)))
        silent_samples = torch.sum(abs_wf < silence_thresh).item()
        silence_ratio = silent_samples / max(1, waveform.shape[-1])

        duration_sec = waveform.shape[-1] / float(sr_out)

        # Contiguous tensor
        waveform = waveform.contiguous()

        return PreparedAudio(
            waveform=waveform,
            sample_rate=sr_out,
            duration_seconds=duration_sec,
            source_path=file_path,
            checksum=checksum,
            peak=peak,
            rms=rms,
            clipping_ratio=clipping_ratio,
            silence_ratio=silence_ratio,
        )

    def _trim_silence_with_padding(
        self, waveform: torch.Tensor, sr: int
    ) -> torch.Tensor:
        """Trims leading/trailing silence while preserving natural trailing padding."""
        abs_wf = torch.abs(waveform.squeeze(0))
        thresh = torch.max(abs_wf) * (10 ** (self.silence_threshold_db / 20.0))
        non_silent = torch.where(abs_wf >= thresh)[0]

        if len(non_silent) == 0:
            return waveform

        start_idx = max(0, non_silent[0].item())
        end_idx = min(len(abs_wf), non_silent[-1].item() + 1)

        # Retain trailing silence padding (300ms to 600ms)
        padding_samples = int(sr * self.trailing_silence_sec)
        end_idx = min(len(abs_wf), end_idx + padding_samples)

        trimmed = waveform[:, start_idx:end_idx]
        return trimmed
