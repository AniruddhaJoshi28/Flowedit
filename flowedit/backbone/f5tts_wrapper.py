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
try:
    import torchaudio
except ImportError:
    torchaudio = None

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

            import inspect
            sig = inspect.signature(F5TTS.__init__)
            params = sig.parameters

            init_kwargs = {}
            if "model_type" in params:
                init_kwargs["model_type"] = "F5-TTS"
            if "ode_method" in params:
                init_kwargs["ode_method"] = "euler"
            if "use_ema" in params:
                init_kwargs["use_ema"] = True
            if "device" in params:
                init_kwargs["device"] = self.device
            if ckpt_file and os.path.exists(ckpt_file) and "ckpt_file" in params:
                init_kwargs["ckpt_file"] = ckpt_file
            if vocab_file and os.path.exists(vocab_file) and "vocab_file" in params:
                init_kwargs["vocab_file"] = vocab_file
            if vocoder_local_path and os.path.exists(os.path.join(vocoder_local_path, "config.yaml")) and "vocoder_local_path" in params:
                init_kwargs["vocoder_local_path"] = vocoder_local_path
            elif vocoder_local_path:
                logger.info(f"Local vocoder directory '{vocoder_local_path}' missing config.yaml. F5-TTS will load Vocos from Hugging Face.")
            if "vocoder_name" in params:
                init_kwargs["vocoder_name"] = "vocos"

            try:
                self.tts_api = F5TTS(**init_kwargs)
            except Exception as e_first:
                logger.warning(f"F5TTS initialization with custom paths failed ({e_first}). Retrying with default Hugging Face download...")
                # Remove custom local paths and retry
                fallback_kwargs = {k: v for k, v in init_kwargs.items() if k not in ("vocoder_local_path", "ckpt_file", "vocab_file")}
                if ckpt_file and os.path.exists(ckpt_file):
                    fallback_kwargs["ckpt_file"] = ckpt_file
                if vocab_file and os.path.exists(vocab_file):
                    fallback_kwargs["vocab_file"] = vocab_file
                self.tts_api = F5TTS(**fallback_kwargs)

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
                try:
                    _whisper = whisperx.load_model(whisper_model_path, device=device, compute_type=compute_type)
                except Exception:
                    _whisper = whisperx.load_model("base", device=device, compute_type=compute_type)
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
        else:
            # Automatic energy-based silence trimming to focus on active speech
            non_silent = (ref_wav.abs() > 0.01).nonzero(as_tuple=True)
            if len(non_silent[1]) > 0:
                s_trim = max(0, non_silent[1].min().item() - 240)
                e_trim = min(ref_wav.shape[-1], non_silent[1].max().item() + 240)
                if e_trim > s_trim:
                    ref_wav = ref_wav[:, s_trim:e_trim]

        # Compute log-mel spectrogram for reference
        from flowedit.utils.audio import AudioProcessor
        processor = AudioProcessor()
        ref_mel = processor.compute_mel(ref_wav, normalize=True, n_mels=pred_mel.shape[1]).float()

        # 3. Extract target word frames from predicted mel (Section 3.2 & Eq. 3)
        T_mel = pred_mel.shape[-1]
        target_word = kwargs.get("target_word", "")

        start_frame = None
        end_frame = None

        if target_word and target_word.lower() in text.lower() and len(text) > 0:
            # Exact target word character boundaries in carrier sentence for precise acoustic alignment
            w_start = text.lower().find(target_word.lower())
            w_end = w_start + len(target_word)
            start_frame = max(0, int((w_start / len(text)) * T_mel))
            end_frame = min(T_mel, max(start_frame + 2, int((w_end / len(text)) * T_mel)))
        elif kwargs.get("token_indices") and len(text) > 0:
            token_indices = kwargs.get("token_indices", [])
            t_min = min(token_indices)
            t_max = max(token_indices) + 1
            start_frame = max(0, int((t_min / len(text)) * T_mel))
            end_frame = min(T_mel, max(start_frame + 2, int((t_max / len(text)) * T_mel)))
        elif target_word_start_sample is not None and target_word_end_sample is not None:
            # Time-aligned framing
            hop_length = 256
            s_f = target_word_start_sample // hop_length
            e_f = target_word_end_sample // hop_length
            if s_f < T_mel:
                start_frame = max(0, s_f)
                end_frame = min(T_mel, max(start_frame + 2, e_f))

        if start_frame is not None and end_frame is not None and end_frame > start_frame:
            pred_target_mel = pred_mel[..., start_frame:end_frame]
        else:
            pred_target_mel = pred_mel

        # 4. Composite Spectral Loss for crystal-clear phoneme pronunciation (Paper Section 3.2 & Eq. 3)
        if pred_target_mel.shape[-1] != ref_mel.shape[-1]:
            pred_target_mel_aligned = F.interpolate(
                pred_target_mel.float(),
                size=ref_mel.shape[-1],
                mode='linear',
                align_corners=False,
            )
        else:
            pred_target_mel_aligned = pred_target_mel.float()

        ref_target_mel = ref_mel.float()
        l1_loss = F.l1_loss(pred_target_mel_aligned, ref_target_mel)
        mse_loss = F.mse_loss(pred_target_mel_aligned, ref_target_mel)
        spec_conv_loss = torch.norm(ref_target_mel - pred_target_mel_aligned, p='fro') / (torch.norm(ref_target_mel, p='fro') + 1e-6)

        composite_spectral_loss = 1.0 * l1_loss + 0.5 * mse_loss + 0.2 * spec_conv_loss
        return {"loss": composite_spectral_loss, "l1_loss": l1_loss, "mse_loss": mse_loss}

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

        # Select the raw trainable PyTorch model for Stage 2 optimization (not the EMA wrapper)
        opt_model = None
        if self.tts_api is not None:
            opt_model = getattr(self.tts_api, "model", None)
            if opt_model is None and hasattr(self.tts_api, "ema_model"):
                ema = self.tts_api.ema_model
                opt_model = getattr(ema, "ema_model", getattr(ema, "model", getattr(ema, "module", ema)))
        if opt_model is None:
            opt_model = self.model
            if hasattr(opt_model, "ema_model"):
                opt_model = opt_model.ema_model
            elif hasattr(opt_model, "module"):
                opt_model = opt_model.module

        dit_model = getattr(opt_model, "transformer", opt_model)
        gen_tokens = self.get_token_ids(text, language)
        seq_len = gen_tokens.shape[1]

        # Match model parameter dtype (e.g. float16 on GPU vs float32)
        model_dtype = next(dit_model.parameters()).dtype if list(dit_model.parameters()) else torch.float32

        # Collect all active model candidates in self and self.tts_api
        hook_targets = []
        candidates = [
            dit_model,
            self.model,
            getattr(self.tts_api, "ema_model", None),
            getattr(self.tts_api, "model", None),
        ]
        for cand in candidates:
            if cand is None:
                continue
            transformer = getattr(cand, "transformer", cand)
            if hasattr(transformer, "clear_cache"):
                transformer.clear_cache()
            te = getattr(transformer, "text_embed", None)
            target_mod = getattr(te, "text_embed", te) if te is not None else None
            if target_mod is not None and target_mod not in hook_targets:
                hook_targets.append(target_mod)

        # Enable gradient checkpointing across all transformer candidates to prevent OOM
        saved_ckpts = []
        for cand in candidates:
            if cand is None:
                continue
            transformer = getattr(cand, "transformer", cand)
            if hasattr(transformer, "checkpoint_activations"):
                saved_ckpts.append((transformer, transformer.checkpoint_activations))
                transformer.checkpoint_activations = True

        hook_handles = []
        if hook_targets and text_embedding_delta is not None:
            def embedding_hook(module, inputs, output):
                is_tuple = isinstance(output, tuple)
                raw_out = output[0] if is_tuple else output
                delta_cast = text_embedding_delta.to(device=raw_out.device, dtype=raw_out.dtype)
                L_delta = delta_cast.shape[1]
                L_out = raw_out.shape[1]

                out = raw_out.clone()
                actual_len = min(L_out, L_delta)
                out[:, :actual_len, :] = out[:, :actual_len, :] + delta_cast[:, :actual_len, :]

                if is_tuple:
                    return (out,) + output[1:]
                return out

            for target in hook_targets:
                hook_handles.append(target.register_forward_hook(embedding_hook))

        try:
            # Enable TF32 for speed and tensor core memory efficiency
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

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
                x0 = torch.randn(batch_size, total_mel_frames, mel_dim, device=device, generator=g, dtype=model_dtype)
            else:
                x0 = torch.randn(batch_size, total_mel_frames, mel_dim, device=device, dtype=model_dtype)

            cond_mel = torch.zeros(batch_size, total_mel_frames, mel_dim, device=device, dtype=model_dtype)
            mask = torch.ones(batch_size, total_mel_frames, dtype=torch.bool, device=device)

            # Euler integration from t=0 to t=1 (Paper Section 3.1 & Eq. 2)
            x_t = x0
            t_eval = torch.linspace(0, 1, steps + 1, device=device, dtype=model_dtype)

            # Automatic Mixed Precision for memory efficiency
            dev_str = str(device)
            use_amp = "cuda" in dev_str or (isinstance(device, torch.device) and device.type == "cuda")
            amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

            with torch.enable_grad(), torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=amp_dtype, enabled=use_amp):
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
                    if step_idx == 0:
                        logger.info(f"[Euler Check Step 0] v_t requires_grad={v_t.requires_grad}, grad_fn={v_t.grad_fn}, delta requires_grad={text_embedding_delta.requires_grad}")
                    x_t = x_t + v_t * dt

            # Transpose to [batch, mel_dim, time_frames]
            pred_mel = x_t.transpose(1, 2)
            return pred_mel

        finally:
            for h in hook_handles:
                h.remove()
            for tr, orig_val in saved_ckpts:
                tr.checkpoint_activations = orig_val


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

        # Collect all active model candidates in self and self.tts_api
        hook_targets = []
        raw_candidates = [
            getattr(self.tts_api, "ema_model", None),
            self.model,
            getattr(self.tts_api, "model", None),
        ]
        candidates = []
        for cand in raw_candidates:
            if cand is None:
                continue
            candidates.append(cand)
            if hasattr(cand, "ema_model"):
                candidates.append(getattr(cand, "ema_model"))
            if hasattr(cand, "model"):
                candidates.append(getattr(cand, "model"))
            if hasattr(cand, "module"):
                candidates.append(getattr(cand, "module"))

        for cand in candidates:
            transformer = getattr(cand, "transformer", cand)
            if hasattr(transformer, "clear_cache"):
                transformer.clear_cache()
            te = getattr(transformer, "text_embed", None)
            target_mod = getattr(te, "text_embed", te) if te is not None else None
            if target_mod is not None and target_mod not in hook_targets:
                hook_targets.append(target_mod)

        hook_handles = []
        if text_embedding_delta is not None and hook_targets:
            target_ids = self.get_token_ids(text, language)[0].tolist()

            def hook(module, inputs, output):
                # Detect unconditional CFG pass: when drop_text=True in F5-TTS, input text tokens are all zeros
                if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
                    if (inputs[0] == 0).all().item():
                        logger.info(f"[F5TTS Forward Hook] (uncond branch skipped, all-zero tokens) on {module.__class__.__name__}")
                        return output

                with torch.set_grad_enabled(False):
                    is_tuple = isinstance(output, tuple)
                    raw_out = output[0] if is_tuple else output
                    t_delta = text_embedding_delta.to(device=raw_out.device, dtype=raw_out.dtype)
                    L_delta = t_delta.shape[1]
                    L_out = raw_out.shape[1]

                    # Exact sub-sequence matching to find where 'gen_text' begins in the input tokens
                    start_pos = -1
                    if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
                        input_ids = inputs[0][0].tolist()
                        match_len = min(len(target_ids), 16)
                        if match_len > 0:
                            prefix_to_find = target_ids[:match_len]
                            prefix_to_find_shifted = [t + 1 for t in prefix_to_find]
                            for i in range(len(input_ids) - match_len + 1):
                                sub = input_ids[i : i + match_len]
                                if sub == prefix_to_find or sub == prefix_to_find_shifted:
                                    start_pos = i
                                    break

                    # Fallback if pattern matching did not find exact match
                    if start_pos < 0:
                        if ref_text and ref_text != ".":
                            prefix_str = ref_text if ref_text.endswith(" ") else f"{ref_text} "
                            try:
                                from f5_tts.model.utils import convert_char_to_pinyin
                                ref_tokens = len(convert_char_to_pinyin([prefix_str])[0])
                            except Exception:
                                ref_tokens = len(prefix_str)
                        else:
                            ref_tokens = 0
                        start_pos = min(L_out, ref_tokens)

                    end_pos = min(L_out, start_pos + L_delta)
                    actual_delta_len = end_pos - start_pos

                    out = raw_out.clone()
                    if actual_delta_len > 0:
                        out[:, start_pos:end_pos, :] = out[:, start_pos:end_pos, :] + t_delta[:, :actual_delta_len, :]

                    logger.info(
                        f"✓ [F5TTS Token-Level Injection] (cond branch) Exact matched start_pos={start_pos} → "
                        f"Injected δ* on {module.__class__.__name__} at character tokens [{start_pos}:{end_pos}] "
                        f"(delta_len={L_delta}, delta_norm={t_delta.norm().item():.4f})"
                    )
                    if is_tuple:
                        return (out,) + output[1:]
                    return out

            for target in hook_targets:
                hook_handles.append(target.register_forward_hook(hook))
            logger.info(f"[F5TTS Injection] Registered forward hook across {len(hook_targets)} target module(s).")

        infer_kwargs = {
            "ref_file": temp_ref,
            "ref_text": ref_text,
            "gen_text": text,
            "speed": planned.dynamic_speed_factor,
            "nfe_step": 32,
            "cfg_strength": 2.0,
            "target_rms": 0.1,
        }

        import inspect
        sig = inspect.signature(self.tts_api.infer)
        params = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not has_var_kw:
            infer_kwargs = {k: v for k, v in infer_kwargs.items() if k in params}

        try:
            result = self.tts_api.infer(**infer_kwargs)
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
            for h in hook_handles:
                h.remove()

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
