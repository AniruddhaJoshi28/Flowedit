"""
F5-TTS Backbone Wrapper for FlowEdit.

This wraps the continuous Flow-Matching F5-TTS model (Diffusion Transformer).
Uses the official high-level F5TTS API class which correctly handles
model loading, tokenizer/vocab resolution, vocoder setup, and inference.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
import tempfile

from flowedit.config import BackboneConfig
from flowedit.backbone.base import TTSBackbone, OptimizationMode, BackboneCapabilities
from flowedit.audio.prompt_validator import validate_reference_audio, ReferenceAudioError
from flowedit.text.duration_planner import DurationPlanner

logger = logging.getLogger(__name__)



class F5TTSBackbone(TTSBackbone):
    """
    F5-TTS wrapper using the official high-level F5TTS API class.
    This ensures correct tokenizer, vocab, model, and vocoder wiring.
    """

    def __init__(self, config: BackboneConfig):
        super().__init__(config)
        self.device_name = config.device
        self.model = None       # raw DiT model reference (for embedding hooks)
        self.vocoder = None     # vocoder reference
        self.tokenizer_instance = None   # vocab char map
        self.tts_api = None     # high-level F5TTS instance
        self._embedding_hook_handle = None

    @property
    def device(self) -> str:
        return self.device_name

    @property
    def tokenizer(self):
        return self.tokenizer_instance

    @property
    def capabilities(self) -> BackboneCapabilities:
        """Return F5-TTS explicit capability contract."""
        return BackboneCapabilities(
            supports_zero_shot=True,
            supports_cross_lingual=True,
            supports_explicit_duration=True,
            supports_speed_control=True,
            supports_phonemes=False,
            supports_differentiable_synthesis=True,
            supports_conditioning_edit=True,
            native_sample_rate=24000,
        )

    @property
    def optimization_mode(self) -> OptimizationMode:
        return OptimizationMode.FLOW_MATCHING


    def tokenize(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Tokenize text into character token IDs."""
        if self.tokenizer_instance and hasattr(self.tokenizer_instance, "encode"):
            ids = self.tokenizer_instance.encode(text, lang=language)
            return {"token_ids": torch.tensor(ids, dtype=torch.long), "raw_tokens": list(text)}
        return {"token_ids": torch.tensor([ord(c) for c in text], dtype=torch.long), "raw_tokens": list(text)}

    def detokenize(self, token_ids: torch.Tensor) -> str:
        """Decode character token IDs back to text."""
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.squeeze().tolist()
        else:
            ids = list(token_ids)
        if self.tokenizer_instance and hasattr(self.tokenizer_instance, "decode"):
            return self.tokenizer_instance.decode(ids)
        return "".join([chr(i) if 0 <= i < 0x10FFFF else "" for i in ids])

    def synthesize(
        self,
        text: str,
        speaker_conditioning: Dict[str, Any],
        language: str = "en",
        user_ref_text: Optional[str] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Direct synthesis without hooks."""
        return self.synthesize_direct(text, speaker_conditioning, language=language, user_ref_text=user_ref_text)



    @property
    def embedding_dim(self) -> int:
        """Return the dimension of the text embeddings."""
        if self.model is not None:
            try:
                sample_emb = self.encode_text("a")
                return sample_emb.shape[-1]
            except Exception:
                pass
        return 512

    def load_model(self):
        """Load the actual F5-TTS model via the high-level API."""
        logger.info("Loading F5-TTS via high-level API (auto-downloads weights on first run)...")
        try:
            from f5_tts.api import F5TTS

            # Try different constructor signatures for different f5-tts versions
            try:
                # Newer versions (v0.3+)
                self.tts_api = F5TTS(
                    model_type="F5-TTS",
                    ckpt_file="",
                    vocab_file="",
                    ode_method="euler",
                    use_ema=True,
                    vocoder_name="vocos",
                    device=self.device,
                )
            except TypeError:
                try:
                    # Older versions without model_type/device
                    self.tts_api = F5TTS(
                        ckpt_file="",
                        vocab_file="",
                        ode_method="euler",
                        use_ema=True,
                        vocoder_name="vocos",
                    )
                except TypeError:
                    # Minimal fallback
                    self.tts_api = F5TTS()

            # Store internal references for Hopfield Memory embedding access
            self.model = getattr(self.tts_api, "ema_model", getattr(self.tts_api, "model", None))
            self.vocoder = getattr(self.tts_api, "vocoder", None)

            # Build a tokenizer wrapper with an .encode() method
            # The aligner expects tokenizer.encode(text, lang=...) → list[int]
            vocab_map = getattr(self.tts_api, "vocab_char_map", None)
            if vocab_map and isinstance(vocab_map, dict):
                self.tokenizer_instance = self._make_char_tokenizer(vocab_map)
            else:
                self.tokenizer_instance = self._make_char_tokenizer(None)

            logger.info("F5-TTS loaded successfully via high-level API.")
        except ImportError:
            logger.error("f5-tts is not installed! Run: pip install f5-tts")
            raise

    @staticmethod
    def _make_char_tokenizer(vocab_map: Optional[dict]):
        """Create a tokenizer wrapper with an .encode() method for the aligner."""

        class CharTokenizer:
            def __init__(self, char_map):
                self.char_map = char_map
                # Build reverse map for decode()
                if char_map:
                    self.id_to_char = {v: k for k, v in char_map.items()}
                else:
                    self.id_to_char = None

            def encode(self, text, lang=None):
                """Map each character to its vocab ID."""
                if self.char_map:
                    return [[self.char_map.get(ch, 0) for ch in text]]
                else:
                    return [[i for i in range(len(text))]]

            def decode(self, token_ids):
                """Reconstruct text from token IDs.
                
                F5-TTS uses character-level tokenization, so each
                token ID maps back to a single character.
                """
                if isinstance(token_ids, torch.Tensor):
                    token_ids = token_ids.tolist()
                if isinstance(token_ids[0], list):
                    token_ids = token_ids[0]
                if self.id_to_char:
                    return "".join(self.id_to_char.get(tid, "?") for tid in token_ids)
                else:
                    # Fallback: token ID == character index, but we don't
                    # know the original text. Return placeholder per token.
                    return "?" * len(token_ids)

        return CharTokenizer(vocab_map)

    def _ensure_loaded(self):
        if self.tts_api is None:
            self.load_model()

    def get_token_ids(self, text: str, language: str = "en") -> torch.Tensor:
        """Tokenize text into IDs."""
        self._ensure_loaded()
        vocab_map = getattr(self.tts_api, "vocab_char_map", None)
        if vocab_map and isinstance(vocab_map, dict):
            # Convert text to list of characters (or tokens if BPE)
            # F5-TTS uses a custom char/pinyin mapping often
            tokens = [vocab_map.get(ch, 0) for ch in text]
            return torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            
        # Fallback if vocab is missing
        tokens = torch.zeros((1, len(text)), dtype=torch.long, device=self.device)
        return tokens

    def encode_text(self, text: str, language: str = "en") -> torch.Tensor:
        """
        Encode text into continuous embeddings.
        Used by Hopfield Memory for key computation.
        """
        self._ensure_loaded()
        tokens = self.get_token_ids(text, language)
        
        # In F5-TTS, the model is often a CFM wrapper around the DiT transformer
        dit_model = getattr(self.model, "transformer", self.model)
        
        with torch.no_grad():
            if hasattr(dit_model, "text_embed"):
                # F5-TTS text_embed expects (tokens, seq_len) where seq_len is 1D tensor
                seq_lengths = torch.tensor([tokens.shape[1]], dtype=torch.long, device=self.device)
                try:
                    embeddings = dit_model.text_embed(tokens, seq_lengths)
                except Exception:
                    # Fallback if signature is different
                    embeddings = dit_model.text_embed(tokens, torch.zeros_like(tokens))
                    
                if isinstance(embeddings, tuple):
                    embeddings = embeddings[0]
            else:
                # If we cannot find text_embed, use a fallback embedding table
                if not hasattr(self, "_fallback_embed"):
                    self._fallback_embed = nn.Embedding(10000, self.embedding_dim).to(self.device)
                embeddings = self._fallback_embed(tokens % 10000)
                
        return embeddings

    def compute_optimization_loss(
        self,
        perturbed_embeddings: torch.Tensor,
        ref_audio_path: str,
        speaker_conditioning: Dict[str, Any],
        text: str,
        language: str = "en",
        target_word_start_time: Optional[float] = None,
        target_word_end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Synthesize from perturbed embeddings and return mel reconstruction loss."""
        pred_out = self.synthesize_from_embeddings(
            text_embeddings=perturbed_embeddings,
            speaker_conditioning=speaker_conditioning,
            text=text,
            language=language,
        )
        pred_waveform = pred_out[0] if isinstance(pred_out, tuple) else pred_out
        pred_waveform = pred_waveform.to(self.device)

        from flowedit.utils.audio import AudioProcessor
        from flowedit.utils.metrics import compute_mel_loss
        
        ap = AudioProcessor()
        ref_waveform, _ = ap.load_audio(ref_audio_path)
        ref_waveform = ref_waveform.to(self.device)
        ref_mel = ap.compute_mel(ref_waveform)

        pred_mel = ap.compute_mel(pred_waveform)
        loss = compute_mel_loss(pred_mel, ref_mel)

        return {
            "loss": loss,
            "mel_loss": loss.item(),
            "embedding_norm": torch.norm(perturbed_embeddings).item()
        }

    def get_speaker_embedding(self, audio_path: Optional[str] = None, language: str = "en", ref_text: Optional[str] = None) -> Dict[str, Any]:
        """Store reference audio path for F5-TTS inference and transcribe if ref_text is missing."""
        self._ensure_loaded()
        
        # Enforce strict reference audio validation — NO SILENT DEMO FALLBACK
        prepared_audio = validate_reference_audio(audio_path, target_sample_rate=24000)

        if not hasattr(self, "_speaker_cache"):
            self._speaker_cache = {}

        cache_key = f"{prepared_audio.checksum}_{language}_{ref_text}"
        if cache_key in self._speaker_cache:
            return self._speaker_cache[cache_key]
        
        import soundfile as sf
        
        # Create a persistent temp file for the processed 24kHz audio
        processed_fd, processed_path = tempfile.mkstemp(suffix=".wav")
        os.close(processed_fd)
        
        sf.write(processed_path, prepared_audio.waveform.squeeze(0).cpu().numpy(), 24000)
        
        prompt_text = ref_text
        if not prompt_text:
            try:
                import whisper
                _whisper = whisper.load_model("base", device=str(self.device))
                ref_result = _whisper.transcribe(prepared_audio.source_path, language=language)
                prompt_text = ref_result.get("text", "").strip()
            except Exception as e:
                logger.warning(f"Auto transcription failed for {audio_path}: {e}")
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


    def synthesize_from_embeddings(
        self,
        text_embeddings: torch.Tensor,
        speaker_conditioning: Dict[str, str],
        text: str,
        language: str = "en",
        solver_method: str = "euler"
    ) -> Tuple[torch.Tensor, int]:
        """Differentiable synthesis using custom ODE solver and Adjoint method.
        
        Bypasses the non-differentiable `tts_api.infer` integration loop by dynamically
        patching the DiT forward pass to run `torchdiffeq.odeint_adjoint` internally,
        while letting the official API handle masking, tokenization, and vocoding.
        """
        self._ensure_loaded()
        logger.info(f"Synthesizing using Differentiable ODE solver (Adjoint Sensitivity with {solver_method})...")

        import types
        import inspect
        try:
            from torchdiffeq import odeint_adjoint
        except ImportError:
            raise ImportError("torchdiffeq is required for the Adjoint Sensitivity Method. Run: pip install torchdiffeq")

        model_obj = getattr(self.tts_api, "ema_model", getattr(self.tts_api, "model", None))
        dit_model = getattr(model_obj, "transformer", model_obj)
        orig_forward = dit_model.forward
        
        # We need the vocoder to return the waveforms, so hook it.
        grabbed_waveforms = []
        vocoder_hook_handle = None
        orig_decode = None
        vocoder_obj = getattr(self.tts_api, "vocoder", getattr(self, "vocoder", None))
        if vocoder_obj is not None:
            def vocoder_hook(module, inputs, output):
                grabbed_waveforms.append(output)
            vocoder_hook_handle = vocoder_obj.register_forward_hook(vocoder_hook)
            if hasattr(vocoder_obj, "decode"):
                orig_decode = vocoder_obj.decode
                def wrapped_decode(*args, **kwargs):
                    output = orig_decode(*args, **kwargs)
                    grabbed_waveforms.append(output)
                    return output
                vocoder_obj.decode = wrapped_decode

        # --- The ODE Interceptor ---
        # We intercept the VERY FIRST call to the transformer inside F5-TTS's solver loop.
        class ODEFV(nn.Module):
            def __init__(self, original_forward, captured_args, captured_kwargs, t_arg_name):
                super().__init__()
                self.original_forward = original_forward
                self.captured_args = captured_args
                self.captured_kwargs = captured_kwargs
                self.t_arg_name = t_arg_name
                self.sig = inspect.signature(original_forward)
                
            def forward(self, t, x):
                t_batch = t.expand(x.shape[0]).to(x.device, dtype=x.dtype)
                bound = self.sig.bind(x, *self.captured_args, **self.captured_kwargs)
                bound.apply_defaults()
                bound.arguments[self.t_arg_name] = t_batch
                return self.original_forward(*bound.args, **bound.kwargs)

        class ODEWrapper(nn.Module):
            def __init__(self, original_forward, steps=32, method="euler"):
                super().__init__()
                self.original_forward = original_forward
                self.steps = steps
                self.method = method
                self.intercepted = False
                
            def forward(self, x, *args, **kwargs):
                if self.intercepted:
                    # After our one big leap, if F5-TTS somehow calls again, just pass it through
                    return self.original_forward(x, *args, **kwargs)
                
                self.intercepted = True
                
                # Dynamically find the time argument
                sig = inspect.signature(self.original_forward)
                t_arg_name = None
                for name in sig.parameters.keys():
                    if name in ['t', 'time', 'timestep']:
                        t_arg_name = name
                        break
                if t_arg_name is None:
                    raise RuntimeError("Could not identify the time argument in DiT signature.")
                
                # Run the full integration loop using adjoint method
                odefunc = ODEFV(self.original_forward, args, kwargs, t_arg_name)
                t_eval = torch.linspace(0, 1, self.steps + 1, device=x.device, dtype=x.dtype)
                
                # odeint_adjoint returns trajectory. We take the final state.
                final_x = odeint_adjoint(odefunc, x, t_eval, method=self.method)[-1]
                
                # Return the delta, so when the outer F5-TTS Euler step does x0 + v * 1.0,
                # it results exactly in final_x.
                return final_x - x

        wrapper = ODEWrapper(orig_forward, steps=32, method=solver_method)
        dit_model.forward = wrapper.forward

        # Prepare audio & text
        import soundfile as sf
        import torchaudio

        temp_ref = speaker_conditioning.get("processed_audio_path")
        if not temp_ref or not os.path.exists(temp_ref):
            temp_ref = speaker_conditioning.get("audio_path", "")

        _orig_load = torchaudio.load
        def _sf_load(path, *args, **kwargs):
            data, sample_rate = sf.read(str(path))
            tensor = torch.from_numpy(data.copy()).float()
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            else:
                tensor = tensor.T
            return tensor, sample_rate
        torchaudio.load = _sf_load

        # The Text Embedding Hook MUST stay, because FlowEdit directly edits the embeddings!
        target_embed_module = getattr(dit_model, "text_embed", None)
        hook_handle = None
        if target_embed_module is not None:
            def hook(module, inputs, output):
                out_tensor = output[0] if isinstance(output, tuple) else output
                T_mel = out_tensor.shape[1]
                L_gen = text_embeddings.shape[1]
                start_pos = max(0, T_mel - L_gen)
                t_embed = text_embeddings.to(device=out_tensor.device, dtype=out_tensor.dtype)
                
                new_out_tensor = out_tensor.clone()
                avail = min(L_gen, T_mel - start_pos)
                new_out_tensor[:, start_pos:start_pos + avail, :] = t_embed[:, :avail, :]
                
                if isinstance(output, tuple):
                    return (new_out_tensor,) + output[1:]
                return new_out_tensor
            hook_handle = target_embed_module.register_forward_hook(hook)

        try:
            ref_text = speaker_conditioning.get("text", ".").strip()
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

            # We DO NOT unwrap `infer`. We call it directly so it handles masking and tokenization.
            # We MUST set nfe_step=1 so that the outer solver takes exactly 1 step (dt=1.0)
            # from 0 to 1, effectively becoming a pass-through for our Adjoint solver.
            with torch.set_grad_enabled(True):
                try:
                    self.tts_api.infer(
                        ref_file=temp_ref,
                        ref_text=ref_text,
                        gen_text=text,
                        speed=planned.dynamic_speed_factor,
                        nfe_step=1,  # CRITICAL: Forces 1 outer step.
                        cfg_strength=2.0,
                        target_rms=0.1,
                        remove_ref=True,
                    )
                except TypeError:
                    self.tts_api.infer(
                        ref_file=temp_ref,
                        ref_text=ref_text,
                        gen_text=text,
                        speed=planned.dynamic_speed_factor,
                        nfe_step=1,  # CRITICAL: Forces 1 outer step.
                        cfg_strength=2.0,
                        target_rms=0.1,
                    )

            if grabbed_waveforms:
                waveform = torch.cat(grabbed_waveforms, dim=-1)
            else:
                raise RuntimeError("Failed to intercept vocoder output tensor.")

            return waveform, 24000

        finally:
            if hook_handle is not None:
                hook_handle.remove()
            if vocoder_hook_handle is not None:
                vocoder_hook_handle.remove()
            if orig_decode is not None and vocoder_obj is not None:
                vocoder_obj.decode = orig_decode
            dit_model.forward = orig_forward
            torchaudio.load = _orig_load
        

                
    def synthesize_direct(
        self,
        text: str,
        speaker_conditioning: Dict[str, str],
        language: str = "en",
        user_ref_text: Optional[str] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Direct F5-TTS synthesis WITHOUT embedding injection hooks.

        Used when no Hopfield corrections are active (or corrections don't
        change the embeddings).  This calls the native F5-TTS inference loop
        without any forward-hooks on TextEmbedding, so the model processes
        text normally and avoids the word-repetition / hallucination artefact
        caused by replacing context-aware embeddings with context-free ones.

        Args:
            text: The gen_text to synthesize.
            speaker_conditioning: Dict from get_speaker_embedding().
            language: Language code.
            user_ref_text: If the user explicitly provided a ref_text via the
                API, pass it here.  When None, Whisper's auto-transcription is
                used but overlapping words are dynamically stripped.
        """
        self._ensure_loaded()
        logger.info("Synthesizing directly via F5-TTS (no embedding hooks)...")

        import soundfile as sf
        import torchaudio

        # --- Prepare reference audio path ---
        temp_ref = speaker_conditioning.get("processed_audio_path")
        if not temp_ref or not os.path.exists(temp_ref):
            temp_ref = speaker_conditioning.get("audio_path", "")

        _orig_load = torchaudio.load
        def _sf_load(path, *args, **kwargs):
            data, sample_rate = sf.read(str(path))
            tensor = torch.from_numpy(data.copy()).float()
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            else:
                tensor = tensor.T
            return tensor, sample_rate
        torchaudio.load = _sf_load

        # --- Vocoder hook (to capture waveform tensors from ALL batches) ---
        # F5-TTS splits long text into multiple batches and calls the vocoder
        # separately for each.  We must collect every batch and concatenate.
        grabbed_waveforms = []
        vocoder_hook_handle = None
        orig_decode = None
        vocoder_obj = getattr(self.tts_api, "vocoder", getattr(self, "vocoder", None))
        if vocoder_obj is not None:
            def vocoder_hook(module, inputs, output):
                grabbed_waveforms.append(output)
            vocoder_hook_handle = vocoder_obj.register_forward_hook(vocoder_hook)

            if hasattr(vocoder_obj, "decode"):
                orig_decode = vocoder_obj.decode
                def wrapped_decode(*args, **kwargs):
                    output = orig_decode(*args, **kwargs)
                    grabbed_waveforms.append(output)
                    return output
                vocoder_obj.decode = wrapped_decode

        try:
            # ── Determine ref_text ─────────────────────────────────────────
            # If the user explicitly provided ref_text, trust it as-is.
            # Otherwise, take Whisper's auto-transcription and dynamically
            # strip any words that also appear in gen_text.  F5-TTS
            # internally concatenates (ref_text + gen_text); overlapping
            # words confuse the model's boundary detection and cause it
            # to bleed reference content into the generated audio.
            if user_ref_text:
                ref_text = user_ref_text.strip()
            else:
                ref_text = speaker_conditioning.get("text", ".").strip()
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

            logger.info(f"[Direct] ref_text='{ref_text}', gen_text='{text}', dynamic_speed={planned.dynamic_speed_factor:.3f}")

            # Call F5-TTS inference directly with dynamic speed
            try:
                self.tts_api.infer(
                    ref_file=temp_ref,
                    ref_text=ref_text,
                    gen_text=text,
                    speed=planned.dynamic_speed_factor,
                    nfe_step=32,
                    cfg_strength=2.0,
                    target_rms=0.1,
                    remove_ref=True,
                )
            except TypeError:
                self.tts_api.infer(
                    ref_file=temp_ref,
                    ref_text=ref_text,
                    gen_text=text,
                    speed=planned.dynamic_speed_factor,
                    nfe_step=32,
                    cfg_strength=2.0,
                    target_rms=0.1,
                )


            if grabbed_waveforms:
                # Concatenate all batch outputs along the time axis
                waveform = torch.cat(grabbed_waveforms, dim=-1)
                logger.info(f"[Direct] Captured {len(grabbed_waveforms)} batch(es), "
                            f"total waveform length: {waveform.shape[-1]} samples")
            else:
                raise RuntimeError("Failed to intercept vocoder output tensor.")

            return waveform, 24000

        finally:
            if vocoder_hook_handle is not None:
                vocoder_hook_handle.remove()
            if orig_decode is not None and vocoder_obj is not None:
                vocoder_obj.decode = orig_decode
            torchaudio.load = _orig_load
