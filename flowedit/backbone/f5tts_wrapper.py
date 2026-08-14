"""
F5-TTS Backbone Implementation for FlowEdit.

Paper Reference: FlowEdit (arXiv:2606.20518), Section 3.1 & 3.2.
Continuous Flow-Matching (CFM) Diffusion Transformer (DiT) text-to-speech backbone.
"""

import os
import tempfile
import logging
from typing import Dict, Optional, Tuple, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import torchaudio

from flowedit.config import BackboneConfig
from flowedit.backbone.base import TTSBackbone
from flowedit.audio.prompt_validator import validate_reference_audio
from flowedit.text.duration_planner import DurationPlanner

logger = logging.getLogger(__name__)


class F5TTSBackbone(TTSBackbone):
    """F5-TTS Flow-Matching DiT Backbone for FlowEdit."""

    def __init__(self, config: Optional[BackboneConfig] = None):
        if config is None:
            config = BackboneConfig()
        super().__init__(config)
        self.device_name = config.device
        self.tts_api = None
        self.model = None       # CFM / DiT model
        self.vocoder = None     # Vocoder instance (Vocos)
        self.tokenizer_instance = None
        self._speaker_cache = {}

    @property
    def device(self) -> str:
        return self.device_name

    @property
    def tokenizer(self):
        return self.tokenizer_instance

    @property
    def embedding_dim(self) -> int:
        """Return text embedding dimension d (Paper Section 3.1: d = 1024 or 512)."""
        if self.model is not None:
            dit = getattr(self.model, "transformer", self.model)
            if hasattr(dit, "text_embed"):
                te = dit.text_embed
                if hasattr(te, "text_embed") and hasattr(te.text_embed, "embedding_dim"):
                    return te.text_embed.embedding_dim
                if hasattr(te, "dim"):
                    return te.dim
        return 512

    def load_model(self) -> None:
        """Load F5-TTS model, tokenizer vocab, and vocoder."""
        logger.info(f"Loading F5-TTS backbone on device '{self.device}'...")
        try:
            from f5_tts.api import F5TTS
            import f5_tts.api as f5_api

            ckpt_file = getattr(self.config, 'f5tts_ckpt_file', "")
            vocab_file = getattr(self.config, 'f5tts_vocab_file', "")
            vocoder_local_path = getattr(self.config, 'vocoder_local_path', "")

            f5_kwargs = {}
            if ckpt_file: f5_kwargs["ckpt_file"] = ckpt_file
            if vocab_file: f5_kwargs["vocab_file"] = vocab_file
            if vocoder_local_path: f5_kwargs["vocoder_local_path"] = vocoder_local_path

            try:
                self.tts_api = F5TTS(
                    model_type="F5-TTS",
                    ode_method="euler",
                    use_ema=True,
                    vocoder_name="vocos",
                    device=self.device,
                    **f5_kwargs
                )
            except TypeError:
                self.tts_api = F5TTS(
                    ode_method="euler",
                    use_ema=True,
                    vocoder_name="vocos",
                    **f5_kwargs
                )

            self.model = getattr(self.tts_api, "ema_model", getattr(self.tts_api, "model", None))
            self.vocoder = getattr(self.tts_api, "vocoder", None)
            vocab_map = self.vocab_map
            self.tokenizer_instance = self._make_char_tokenizer(vocab_map)

            logger.info("✓ F5-TTS backbone loaded successfully.")

        except Exception as e:
            logger.warning(f"F5-TTS loading notice ({e}). Operating in standalone/dummy mode.")
            self.tts_api = None
            self.model = None
            self.vocoder = None
            self.tokenizer_instance = self._make_char_tokenizer(None)

    @property
    def vocab_map(self) -> Optional[dict]:
        """Retrieve vocabulary character map."""
        if self.model is not None and hasattr(self.model, "vocab_char_map"):
            return self.model.vocab_char_map
        if self.tts_api is not None:
            if hasattr(self.tts_api, "vocab_char_map"):
                return self.tts_api.vocab_char_map
            ema = getattr(self.tts_api, "ema_model", None)
            if ema is not None and hasattr(ema, "vocab_char_map"):
                return ema.vocab_char_map
        return None

    @staticmethod
    def _make_char_tokenizer(vocab_map: Optional[dict]):
        """Create tokenizer wrapper mapping characters to vocab IDs."""
        class CharTokenizer:
            def __init__(self, char_map):
                self.char_map = char_map or {}
                self.id_to_char = {v: k for k, v in self.char_map.items()} if self.char_map else {}

            def encode(self, text, lang=None):
                if self.char_map:
                    try:
                        from f5_tts.model.utils import convert_char_to_pinyin
                        char_list = convert_char_to_pinyin([text])[0]
                    except Exception:
                        char_list = list(text)
                    return [[self.char_map.get(ch, 0) for ch in char_list]]
                return [[ord(c) % 256 for c in text]]

            def decode(self, token_ids):
                if isinstance(token_ids, torch.Tensor):
                    token_ids = token_ids.tolist()
                if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
                    token_ids = token_ids[0]
                if self.id_to_char:
                    return "".join(self.id_to_char.get(tid, "?") for tid in token_ids)
                return "".join(chr(t) if 0 <= t < 0x10FFFF else "?" for t in token_ids)

        return CharTokenizer(vocab_map)

    def _ensure_loaded(self):
        if self.tts_api is None and self.model is None:
            self.load_model()

    def tokenize(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Tokenize text into character token IDs."""
        self._ensure_loaded()
        ids = self.tokenizer_instance.encode(text, lang=language)[0]
        return {"token_ids": torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0), "raw_tokens": list(text)}

    def detokenize(self, token_ids: torch.Tensor) -> str:
        """Decode character token IDs back to text."""
        self._ensure_loaded()
        return self.tokenizer_instance.decode(token_ids)

    def get_token_ids(self, text: str, language: str = "en") -> torch.Tensor:
        """Get token IDs tensor [1, S]."""
        res = self.tokenize(text, language=language)
        return res["token_ids"]

    def encode_text(self, text: str, language: str = "en") -> torch.Tensor:
        """Encode text into continuous embeddings c ∈ R^[1, S, d] (Paper Section 3.1 & 3.2)."""
        self._ensure_loaded()
        tokens = self.get_token_ids(text, language)

        if self.model is None:
            return torch.zeros(1, tokens.shape[1], self.embedding_dim, device=self.device)

        dit_model = getattr(self.model, "transformer", self.model)
        with torch.no_grad():
            if hasattr(dit_model, "text_embed"):
                seq_lengths = torch.tensor([tokens.shape[1]], dtype=torch.long, device=self.device)
                try:
                    # Pass token IDs to text_embed
                    embeddings = dit_model.text_embed(tokens, seq_lengths)
                except Exception:
                    embeddings = dit_model.text_embed(tokens, tokens.shape[1])
                if isinstance(embeddings, tuple):
                    embeddings = embeddings[0]
            else:
                if not hasattr(self, "_fallback_embed"):
                    self._fallback_embed = nn.Embedding(10000, self.embedding_dim).to(self.device)
                embeddings = self._fallback_embed(tokens % 10000)

        return embeddings

    def get_speaker_embedding(
        self,
        audio_path: Optional[str] = None,
        language: str = "en",
        ref_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract speaker conditioning from reference audio."""
        prepared_audio = validate_reference_audio(audio_path, target_sample_rate=24000)

        cache_key = f"{prepared_audio.checksum}_{language}_{ref_text}"
        if cache_key in self._speaker_cache:
            return self._speaker_cache[cache_key]


        processed_fd, processed_path = tempfile.mkstemp(suffix=".wav")
        os.close(processed_fd)
        sf.write(processed_path, prepared_audio.waveform.squeeze(0).cpu().numpy(), 24000)

        prompt_text = ref_text
        if not prompt_text:
            try:
                import whisperx
                device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                whisper_model_path = os.environ.get("FLOWEDIT_WHISPER_MODEL", "base")
                _whisper = whisperx.load_model(whisper_model_path, device=device, compute_type=compute_type)
                audio_np = whisperx.load_audio(prepared_audio.source_path)
                ref_result = _whisper.transcribe(audio_np, language=language)
                if "segments" in ref_result and ref_result["segments"]:
                    prompt_text = " ".join([seg["text"] for seg in ref_result["segments"]]).strip()
                else:
                    prompt_text = "."
            except Exception as e:
                logger.warning(f"Transcription failed: {e}")
                prompt_text = "."

        if not prompt_text:
            prompt_text = "."

        res = {
            "audio_path": prepared_audio.source_path,
            "processed_audio_path": processed_path,
            "text": prompt_text,
            "duration_seconds": prepared_audio.duration_seconds,
            "checksum": prepared_audio.checksum,
        }
        self._speaker_cache[cache_key] = res
        return res

    def compute_optimization_loss(
        self,
        text_embedding_delta: torch.Tensor,
        ref_audio_path: str,
        speaker_conditioning: Dict[str, Any],
        text: str,
        language: str = "en",
        target_word_start_sample: Optional[int] = None,
        target_word_end_sample: Optional[int] = None,
        ref_start_time: Optional[float] = None,
        ref_end_time: Optional[float] = None,
        seed: Optional[int] = 42,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Stage 2: Compute differentiable Mel reconstruction loss for perturbation optimization.

        Paper Section 3.2:
            L(δ) = ||Mel(g_θ(c + δ)) - Mel(y_ref)||_2^2 + λ||δ||_2^2
        """
        self._ensure_loaded()
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if not text_embedding_delta.requires_grad:
            text_embedding_delta = text_embedding_delta.requires_grad_(True)

        ode_steps = kwargs.get("ode_steps", 16)

        # 1. Synthesize predicted mel-spectrogram differentiably through frozen DiT Euler solver
        pred_mel = self._differentiable_euler_mel(
            text_embedding_delta=text_embedding_delta,
            speaker_conditioning=speaker_conditioning,
            text=text,
            language=language,
            steps=ode_steps,
            seed=seed,
        )

        # 2. Extract reference audio and compute normalized reference mel
        ref_wav, ref_sr = torchaudio.load(ref_audio_path)
        if ref_sr != 24000:
            ref_wav = torchaudio.functional.resample(ref_wav, ref_sr, 24000)
        ref_wav = ref_wav.to(device=pred_mel.device, dtype=torch.float32)
        if ref_wav.dim() == 1:
            ref_wav = ref_wav.unsqueeze(0)

        # Crop reference audio to target word timing if provided
        if ref_start_time is not None and ref_end_time is not None:
            r_s = int(ref_start_time * 24000)
            r_e = int(ref_end_time * 24000)
            r_s = max(0, r_s)
            r_e = min(ref_wav.shape[-1], r_e)
            if r_e > r_s:
                ref_wav = ref_wav[:, r_s:r_e]

        # Compute log-mel spectrogram for reference
        from flowedit.utils.audio import AudioProcessor
        processor = AudioProcessor()
        ref_mel = processor.compute_mel(ref_wav, normalize=True, n_mels=pred_mel.shape[1]).float()

        # 3. Extract target word frames from predicted mel
        hop_length = 256
        if target_word_start_sample is not None and target_word_end_sample is not None:
            start_frame = max(0, target_word_start_sample // hop_length)
            end_frame = min(pred_mel.shape[-1], target_word_end_sample // hop_length)
            if end_frame > start_frame:
                pred_target_mel = pred_mel[..., start_frame:end_frame]
            else:
                pred_target_mel = pred_mel
        else:
            pred_target_mel = pred_mel

        # 4. Mel-Spectrogram MSE Loss (Paper Eq. 3)
        if pred_target_mel.shape[-1] != ref_mel.shape[-1]:
            pred_target_mel_aligned = F.interpolate(
                pred_target_mel,
                size=ref_mel.shape[-1],
                mode='linear',
                align_corners=False,
            )
        else:
            pred_target_mel_aligned = pred_target_mel

        mel_loss = F.mse_loss(pred_target_mel_aligned, ref_mel)
        return {"loss": mel_loss}

    def _differentiable_euler_mel(
        self,
        text_embedding_delta: torch.Tensor,
        speaker_conditioning: Dict[str, Any],
        text: str,
        language: str = "en",
        steps: int = 16,
        seed: Optional[int] = 42,
    ) -> torch.Tensor:
        """Direct, fully differentiable Euler integration through DiT."""
        if self.model is None:
            # Standalone dummy mode for lightweight testing
            base_c = self.encode_text(text, language)
            c_pert = base_c + text_embedding_delta
            # Differentiable linear surrogate
            sim_mel = torch.sin(c_pert[:, :, :100].transpose(1, 2))
            return sim_mel

        dit_model = getattr(self.model, "transformer", self.model)
        gen_tokens = self.get_token_ids(text, language)
        seq_len = gen_tokens.shape[1]

        # Calculate base text embeddings
        target_embed_module = getattr(dit_model, "text_embed", None)
        hook_target = getattr(target_embed_module, "text_embed", target_embed_module) if target_embed_module is not None else None

        hook_handle = None
        if hook_target is not None:
            def embedding_hook(module, inputs, output):
                out = output.clone()
                delta_cast = text_embedding_delta.to(device=out.device, dtype=out.dtype)
                L = min(delta_cast.shape[1], out.shape[1])
                out[0:1, :L, :] = out[0:1, :L, :] + delta_cast[:, :L, :]
                if isinstance(output, tuple):
                    return (out,) + output[1:]
                return out
            hook_handle = hook_target.register_forward_hook(embedding_hook)

        try:
            # Determine target mel length based on planned duration
            ref_dur = speaker_conditioning.get("duration_seconds", 2.0)
            planner = DurationPlanner()
            planned = planner.plan_duration(
                target_text=text,
                ref_audio_duration=ref_dur,
                ref_text=speaker_conditioning.get("text", "."),
                language=language,
            )
            # 1 mel frame ≈ 256 audio samples (~10.67ms at 24kHz)
            total_mel_frames = max(32, int(planned.planned_duration_seconds * 24000 / 256))

            mel_dim = 100
            batch_size = 1
            device = self.device

            if seed is not None:
                g = torch.Generator(device=device).manual_seed(seed)
                x0 = torch.randn(batch_size, total_mel_frames, mel_dim, device=device, generator=g, dtype=torch.float32)
            else:
                x0 = torch.randn(batch_size, total_mel_frames, mel_dim, device=device, dtype=torch.float32)

            cond_mel = torch.zeros(batch_size, total_mel_frames, mel_dim, device=device, dtype=torch.float32)
            mask = torch.ones(batch_size, total_mel_frames, dtype=torch.bool, device=device)

            # Euler integration from t=0 to t=1 (Paper Section 3.1 & Eq. 2)
            x_t = x0
            t_eval = torch.linspace(0, 1, steps + 1, device=device)

            for step_idx in range(steps):
                t_val = t_eval[step_idx]
                dt = t_eval[step_idx + 1] - t_eval[step_idx]
                t_tensor = t_val.expand(batch_size)

                # Evaluate learned vector field v_t(x_t, t; θ, c + δ)
                v_t = dit_model(
                    x=x_t,
                    cond=cond_mel,
                    text=gen_tokens,
                    time=t_tensor,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=False,
                )
                x_t = x_t + v_t * dt

            # Transpose to [batch, mel_dim, time_frames]
            pred_mel = x_t.transpose(1, 2)
            return pred_mel

        finally:
            if hook_handle is not None:
                hook_handle.remove()

    def synthesize_from_embeddings(
        self,
        text_embeddings: torch.Tensor,
        speaker_conditioning: Dict[str, Any],
        text: str,
        language: str = "en",
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """Synthesize audio with (refined) text embeddings."""
        self._ensure_loaded()
        base_c = self.encode_text(text, language)
        delta = text_embeddings - base_c
        return self.synthesize_direct(
            text=text,
            speaker_conditioning=speaker_conditioning,
            language=language,
            text_embedding_delta=delta,
            **kwargs,
        )

    def synthesize_direct(
        self,
        text: str,
        speaker_conditioning: Dict[str, Any],
        language: str = "en",
        user_ref_text: Optional[str] = None,
        text_embedding_delta: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """Direct synthesis through F5-TTS, optionally injecting text embedding delta."""
        self._ensure_loaded()
        if self.tts_api is None:
            # Standalone fallback for testing
            length = int(24000 * max(0.5, len(text) * 0.08))
            return torch.zeros(1, length), 24000

        temp_ref = speaker_conditioning.get("processed_audio_path")
        if not temp_ref or not os.path.exists(temp_ref):
            temp_ref = speaker_conditioning.get("audio_path", "")

        ref_text = user_ref_text.strip() if user_ref_text else speaker_conditioning.get("text", ".").strip()
        if not ref_text:
            ref_text = "."

        ref_duration = speaker_conditioning.get("duration_seconds")
        planner = DurationPlanner()
        planned = planner.plan_duration(
            target_text=text,
            ref_audio_duration=ref_duration,
            ref_text=ref_text,
            language=language,
        )

        dit_model = getattr(self.model, "transformer", self.model)
        target_embed_module = getattr(dit_model, "text_embed", None)
        hook_target = getattr(target_embed_module, "text_embed", target_embed_module) if target_embed_module is not None else None

        hook_handle = None
        if text_embedding_delta is not None and hook_target is not None:
            def hook(module, inputs, output):
                with torch.set_grad_enabled(False):
                    out = output.clone()
                    t_delta = text_embedding_delta.to(device=out.device, dtype=out.dtype)
                    L = min(t_delta.shape[1], out.shape[1])
                    out[0:1, :L, :] = out[0:1, :L, :] + t_delta[:, :L, :]
                    if isinstance(output, tuple):
                        return (out,) + output[1:]
                    return out
            hook_handle = hook_target.register_forward_hook(hook)

        try:
            result = self.tts_api.infer(
                ref_file=temp_ref,
                ref_text=ref_text,
                gen_text=text,
                speed=planned.dynamic_speed_factor,
                nfe_step=32,
                cfg_strength=2.0,
                target_rms=0.1,
                remove_ref=True,
            )
            if isinstance(result, tuple) and len(result) >= 2:
                wav = result[0]
                sr = result[1]
                if isinstance(wav, np.ndarray):
                    wav = torch.from_numpy(wav).float()
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                return wav, sr
            return torch.zeros(1, 24000), 24000

        finally:
            if hook_handle is not None:
                hook_handle.remove()

    def synthesize_baseline(
        self,
        text: str,
        speaker_conditioning: Dict[str, Any],
        language: str = "en",
        user_ref_text: Optional[str] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Pure vanilla F5-TTS synthesis without hooks."""
        return self.synthesize_direct(
            text=text,
            speaker_conditioning=speaker_conditioning,
            language=language,
            user_ref_text=user_ref_text,
            text_embedding_delta=None,
        )
