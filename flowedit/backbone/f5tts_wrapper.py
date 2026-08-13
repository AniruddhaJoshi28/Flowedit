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
            import f5_tts.api as f5_api

            # -- Fallback Dummy Vocoder Patch --
            orig_load_vocoder = getattr(f5_api, "load_vocoder", None)
            
            class DummyVocoder(nn.Module):
                def __init__(self):
                    super().__init__()
                def decode(self, x, **kwargs):
                    logger.warning("DummyVocoder: Outputting silence because vocoder failed to download.")
                    if isinstance(x, tuple):
                        x = x[0]
                    # Generate zero waveform: 1 mel frame ~ 256 audio samples
                    return torch.zeros(1, x.shape[-1] * 256).to(x.device)
                def forward(self, x, **kwargs):
                    return self.decode(x, **kwargs)

            def safe_load_vocoder(*args, **kwargs):
                if orig_load_vocoder is not None:
                    try:
                        return orig_load_vocoder(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"Failed to download/load vocoder: {e}. Using DummyVocoder.")
                        return DummyVocoder()
                return DummyVocoder()
                
            if orig_load_vocoder is not None:
                f5_api.load_vocoder = safe_load_vocoder
            # ----------------------------------

            ckpt_file = getattr(self.config, 'f5tts_ckpt_file', "")
            vocab_file = getattr(self.config, 'f5tts_vocab_file', "")
            vocoder_local_path = getattr(self.config, 'vocoder_local_path', "")
            
            logger.info(f"Initializing F5TTS with ckpt_file='{ckpt_file}', vocab_file='{vocab_file}', vocoder_local_path='{vocoder_local_path}'")
            
            # Prepare kwargs to avoid passing empty strings if they expect None
            f5_kwargs = {}
            if ckpt_file: f5_kwargs["ckpt_file"] = ckpt_file
            if vocab_file: f5_kwargs["vocab_file"] = vocab_file
            if vocoder_local_path: f5_kwargs["vocoder_local_path"] = vocoder_local_path

            try:
                try:
                    try:
                        # Newer versions (v0.3+)
                        self.tts_api = F5TTS(
                            model_type="F5-TTS",
                            ode_method="euler",
                            use_ema=True,
                            vocoder_name="vocos",
                            device=self.device,
                            **f5_kwargs
                        )
                    except TypeError:
                        try:
                            # Older versions without model_type/device
                            self.tts_api = F5TTS(
                                ode_method="euler",
                                use_ema=True,
                                vocoder_name="vocos",
                                **f5_kwargs
                            )
                        except TypeError:
                            # Minimal fallback
                            self.tts_api = F5TTS(**f5_kwargs)
                except Exception as e:
                    logger.error(f"F5TTS failed to download/initialize (network/proxy error?). Error: {e}")
                    logger.warning("F5TTS backbone will be UNAVAILABLE. Synthesis will fail until models are downloaded.")
                    self.tts_api = None
            finally:
                # Restore original to prevent side effects in other parts of the app
                if orig_load_vocoder is not None:
                    f5_api.load_vocoder = orig_load_vocoder

            if self.tts_api is not None:
                # Store internal references for Hopfield Memory embedding access
                self.model = getattr(self.tts_api, "ema_model", getattr(self.tts_api, "model", None))
                self.vocoder = getattr(self.tts_api, "vocoder", None)

                # Build a tokenizer wrapper with an .encode() method
                # The aligner expects tokenizer.encode(text, lang=...) → list[int]
                vocab_map = self.vocab_map
                if vocab_map and isinstance(vocab_map, dict):
                    self.tokenizer_instance = self._make_char_tokenizer(vocab_map)
                else:
                    self.tokenizer_instance = self._make_char_tokenizer(None)

                logger.info("F5-TTS loaded successfully via high-level API.")
            else:
                self.model = None
                self.vocoder = None
                self.tokenizer_instance = self._make_char_tokenizer(None)
                logger.warning("F5-TTS is running in dummy mode. Models were NOT loaded.")

        except ImportError:
            logger.error("f5-tts is not installed! Run: pip install f5-tts")
            raise

    @property
    def vocab_map(self) -> Optional[dict]:
        """Retrieve vocabulary character map from backbone model or tts_api."""
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
                    try:
                        from f5_tts.model.utils import convert_char_to_pinyin
                        char_list = convert_char_to_pinyin([text])[0]
                    except Exception:
                        char_list = list(text)
                    return [[self.char_map.get(ch, 0) for ch in char_list]]
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
        vocab_map = self.vocab_map
        if vocab_map and isinstance(vocab_map, dict):
            try:
                from f5_tts.model.utils import convert_char_to_pinyin
                char_list = convert_char_to_pinyin([text])[0]
            except Exception:
                char_list = list(text)
            tokens = [vocab_map.get(ch, 0) for ch in char_list]
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
        
        if self.model is None:
            logger.warning("encode_text: F5-TTS is in dummy mode. Returning zero embeddings.")
            return torch.zeros(1, tokens.shape[1], self.embedding_dim).to(self.device)
            
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
                import whisperx
                whisper_model_path = os.environ.get("FLOWEDIT_WHISPER_MODEL", "base")
                
                device = "cuda" if str(self.device) == "cuda" or (hasattr(torch.cuda, "is_available") and torch.cuda.is_available()) else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                
                _whisper = whisperx.load_model(whisper_model_path, device=device, compute_type=compute_type)
                audio = whisperx.load_audio(prepared_audio.source_path)
                ref_result = _whisper.transcribe(audio, language=language)
                
                if "segments" in ref_result:
                    prompt_text = " ".join([seg["text"] for seg in ref_result["segments"]]).strip()
                else:
                    prompt_text = "."
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
        text_embedding_delta: torch.Tensor,
        speaker_conditioning: Dict[str, str],
        text: str,
        language: str = "en",
        cfg_strength: float = 0.0,
        solver_method: str = "euler"
    ) -> Tuple[torch.Tensor, int]:
        """Differentiable synthesis using custom ODE solver and Adjoint method.
        
        Bypasses the non-differentiable `tts_api.infer` integration loop by dynamically
        patching the DiT forward pass to run `torchdiffeq.odeint_adjoint` internally,
        while letting the official API handle masking, tokenization, and vocoding.
        """
        self._ensure_loaded()
        if self.tts_api is None:
            logger.warning("synthesize_from_embeddings: F5-TTS is in dummy mode. Returning zero waveform.")
            return torch.zeros(1, 100 * 256), 24000

        logger.info(f"Synthesizing using Differentiable ODE solver (Adjoint Sensitivity with {solver_method})...")

        import types
        import inspect
        try:
            from torchdiffeq import odeint
        except ImportError:
            raise ImportError("torchdiffeq is required for the ODE solver. Run: pip install torchdiffeq")

        model_obj = getattr(self.tts_api, "ema_model", getattr(self.tts_api, "model", None))
        dit_model = getattr(model_obj, "transformer", model_obj)
        orig_forward = dit_model.forward
        
        # Unwrap `model_obj.sample` to bypass `@torch.inference_mode()` if present
        orig_sample = getattr(model_obj, "sample", None)
        if orig_sample is not None:
            target_sample = orig_sample
            if hasattr(target_sample, "__func__"):
                func = target_sample.__func__
                if hasattr(func, "__wrapped__"):
                    target_sample = func.__wrapped__.__get__(model_obj, model_obj.__class__)
            elif hasattr(target_sample, "__wrapped__"):
                target_sample = target_sample.__wrapped__
            model_obj.sample = target_sample
        
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
                
                # Unwrap the decode method to bypass @torch.inference_mode()
                target_decode = orig_decode
                if hasattr(orig_decode, "__func__"):
                    func = orig_decode.__func__
                    if hasattr(func, "__wrapped__"):
                        target_decode = func.__wrapped__.__get__(vocoder_obj, vocoder_obj.__class__)
                elif hasattr(orig_decode, "__wrapped__"):
                    target_decode = orig_decode.__wrapped__
                    
                def wrapped_decode(*args, **kwargs):
                    if len(args) > 0:
                        features_input = args[0]
                    else:
                        features_input = kwargs.get("features_input", kwargs.get("features", None))
                        
                    # Extract L_gen from the actual vocoder input tensor shape
                    if features_input is not None and isinstance(features_input, torch.Tensor):
                        # features_input shape is typically [B, n_mels, T_gen]
                        actual_L_gen = features_input.shape[-1]
                    else:
                        actual_L_gen = 0
                    wrapper_self._saved_vocoder_input_shape = (1, 100, actual_L_gen)
                    
                    # We just run target_decode so F5-TTS doesn't crash, but we will ignore its output
                    with torch.no_grad():
                        output = target_decode(*args, **kwargs)
                    return output
                vocoder_obj.decode = wrapped_decode

        wrapper_self = self
        
        # --- The ODE Interceptor ---
        # We intercept the VERY FIRST call to the transformer inside F5-TTS's solver loop.
        class ODEFV(nn.Module):
            def __init__(self, original_forward, captured_args, captured_kwargs, t_arg_name, cfg_strength=0.0):
                super().__init__()
                self.original_forward = original_forward
                self.captured_args = captured_args
                self.captured_kwargs = captured_kwargs
                self.t_arg_name = t_arg_name
                self.cfg_strength = cfg_strength
                self.sig = inspect.signature(original_forward)
                # Detect the model's parameter dtype (typically float16 for F5-TTS)
                self._model_dtype = None
                if hasattr(original_forward, '__self__'):
                    for p in original_forward.__self__.parameters():
                        self._model_dtype = p.dtype
                        break
                
            def forward(self, t, x):
                # During the adjoint backward pass, x and t may be float32 while
                # model weights are float16. Cast to match the model's dtype.
                # IMPORTANT: Only cast floating-point tensors. Integer tensors (e.g.
                # token IDs for nn.Embedding) must stay as Long/Int.
                model_dtype = self._model_dtype or x.dtype
                x_cast = x.to(dtype=model_dtype)
                t_batch = t.expand(x.shape[0]).to(x.device, dtype=model_dtype)
                
                # Also cast any floating-point tensor args/kwargs to match,
                # and expand batch dim 0 if x has batch size > 1 (e.g. CFG batch size 2).
                cast_args = []
                for arg in self.captured_args:
                    if isinstance(arg, torch.Tensor):
                        t_arg = arg
                        if t_arg.dim() > 0 and t_arg.shape[0] == 1 and x.shape[0] > 1:
                            t_arg = t_arg.expand(x.shape[0], *t_arg.shape[1:])
                        if t_arg.is_floating_point():
                            t_arg = t_arg.to(dtype=model_dtype)
                        cast_args.append(t_arg)
                    else:
                        cast_args.append(arg)

                cast_kwargs = {}
                for k, v in self.captured_kwargs.items():
                    if isinstance(v, torch.Tensor):
                        t_v = v
                        if t_v.dim() > 0 and t_v.shape[0] == 1 and x.shape[0] > 1:
                            t_v = t_v.expand(x.shape[0], *t_v.shape[1:])
                        if t_v.is_floating_point():
                            t_v = t_v.to(dtype=model_dtype)
                        cast_kwargs[k] = t_v
                    else:
                        cast_kwargs[k] = v
                
                bound = self.sig.bind(x_cast, *cast_args, **cast_kwargs)
                bound.apply_defaults()
                bound.arguments[self.t_arg_name] = t_batch
                
                # CRITICAL: Force cache=False to ensure text_embed is evaluated at every step.
                if 'cache' in bound.arguments:
                    bound.arguments['cache'] = False
                    
                result = self.original_forward(*bound.args, **bound.kwargs)
                
                # If CFG is active, DiT returns batch size 2 (cond, uncond), but the ODE state (x) has batch size 1.
                # We MUST combine them here just like cfm.fn does, so the derivative matches the state size!
                if result.shape[0] == 2 and x.shape[0] == 1 and self.cfg_strength > 0:
                    cond, uncond = result.chunk(2, dim=0)
                    result = uncond + self.cfg_strength * (cond - uncond)
                    
                # Cast output back to the ODE solver's expected dtype (x.dtype)
                return result.to(dtype=x.dtype)

        class ODEWrapper(nn.Module):
            def __init__(self, original_forward, steps=32, method="euler", cfg_strength=0.0):
                super().__init__()
                self.original_forward = original_forward
                self.steps = steps
                self.method = method
                self.cfg_strength = cfg_strength
                self.intercepted = False
                
            def forward(self, x, *args, **kwargs):
                
                # Dynamically find the time argument
                sig = inspect.signature(self.original_forward)
                t_arg_name = None
                for name in sig.parameters.keys():
                    if name in ['t', 'time', 'timestep']:
                        t_arg_name = name
                        break
                if t_arg_name is None:
                    raise RuntimeError("Could not identify the time argument in DiT signature.")
                if self.intercepted:
                    exp_args = []
                    for arg in args:
                        if isinstance(arg, torch.Tensor) and arg.dim() > 0 and arg.shape[0] == 1 and x.shape[0] > 1:
                            exp_args.append(arg.expand(x.shape[0], *arg.shape[1:]))
                        else:
                            exp_args.append(arg)
                    exp_kwargs = {}
                    for k, v in kwargs.items():
                        if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == 1 and x.shape[0] > 1:
                            exp_kwargs[k] = v.expand(x.shape[0], *v.shape[1:])
                        else:
                            exp_kwargs[k] = v
                    return self.original_forward(x, *exp_args, **exp_kwargs)
                self.intercepted = True

                # We spawn a new thread to run the ODE solver.
                # PyTorch's inference_mode is thread-local. Since f5_tts puts us inside an
                # unescapable inference_mode context, any tensor created here will be an 
                # inference tensor and cause odeint_adjoint to crash. By running in a new thread,
                # we start with a clean context, allowing us to strip the inference property.
                import concurrent.futures

                def _run_ode_solver():
                    with torch.set_grad_enabled(True):
                        # Clone x to strip its inference property
                        x_norm = x.clone() if isinstance(x, torch.Tensor) else x
                        
                        cloned_args = []
                        for arg in args:
                            if isinstance(arg, torch.Tensor):
                                t_arg = arg.clone()
                                if t_arg.dim() > 0 and t_arg.shape[0] == 1 and isinstance(x_norm, torch.Tensor) and x_norm.shape[0] > 1:
                                    t_arg = t_arg.expand(x_norm.shape[0], *t_arg.shape[1:])
                                cloned_args.append(t_arg)
                            else:
                                cloned_args.append(arg)
                        cloned_args = tuple(cloned_args)
                        
                        cloned_kwargs = {}
                        for k, v in kwargs.items():
                            if isinstance(v, torch.Tensor):
                                t_v = v.clone()
                                if t_v.dim() > 0 and t_v.shape[0] == 1 and isinstance(x_norm, torch.Tensor) and x_norm.shape[0] > 1:
                                    t_v = t_v.expand(x_norm.shape[0], *t_v.shape[1:])
                                cloned_kwargs[k] = t_v
                            else:
                                cloned_kwargs[k] = v
                                
                        t_eval = torch.linspace(0, 1, self.steps, device=x_norm.device, dtype=x_norm.dtype)
                        odefunc = ODEFV(self.original_forward, cloned_args, cloned_kwargs, t_arg_name, self.cfg_strength)
                        
                        res = odeint(
                            odefunc, 
                            x_norm, 
                            t_eval, 
                            method=self.method
                        )[-1]
                        
                        if not hook_fired[0]:
                            raise RuntimeError("The text embedding hook did NOT fire during the ODE solver loop! This means F5-TTS is caching or bypassing 'text_embed'.")
                            
                        return res
                        
                final_x = _run_ode_solver()
                    
                with torch.set_grad_enabled(True):
                    logger.warning(f"[DIAG] odeint_adjoint final_x requires_grad: {final_x.requires_grad}")
                    
                    # Save final_x so we can manually decode it later outside F5-TTS infer's inference_mode!
                    wrapper_self._saved_final_x = final_x
                
                # Return the delta, so when the outer F5-TTS Euler step does x0 + v * 1.0,
                # it results exactly in final_x.
                delta = final_x - x
                
                # If we combined CFG, the delta has batch size 1. But the outer F5-TTS loop (cfm.fn)
                # EXPECTS a batch size 2 tensor so it can do its own chunk and combine.
                # We replicate it to batch size 2 so cfm.fn's combine operation just yields delta again!
                if delta.shape[0] == 1 and getattr(self, "cfg_strength", 0) > 0:
                    delta = torch.cat([delta, delta], dim=0)
                    
                return delta

        wrapper = ODEWrapper(orig_forward, steps=32, method=solver_method, cfg_strength=cfg_strength)
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

        ref_text = speaker_conditioning.get("text", ".").strip()
        if not ref_text:
            ref_text = "."

        ref_tokens = self.get_token_ids(ref_text, language)
        ref_len = ref_tokens.shape[1]
        gen_tokens = self.get_token_ids(text, language)

        target_embed_module = getattr(dit_model, "text_embed", None)
        if target_embed_module is None:
            # Print available modules to help debug
            modules = [name for name, _ in dit_model.named_modules()]
            raise RuntimeError(f"Could not find 'text_embed' in F5-TTS DiT. Available modules: {modules}")

        hook_handle = None
        hook_fired = [False]
        hook_call_count = [0]
        
        def hook(module, inputs, output):
            hook_call_count[0] += 1
            # F5-TTS cfg_infer=True calls text_embed twice sequentially (cond, then uncond).
            # We MUST only apply the learned memory delta to the conditional stream!
            if hook_call_count[0] % 2 == 0:
                return output
                
            hook_fired[0] = True
            with torch.set_grad_enabled(True):
                out_tensor = output.clone()
                t_delta = text_embedding_delta.to(device=out_tensor.device, dtype=out_tensor.dtype)
                
                B = out_tensor.shape[0]
                T_mel = out_tensor.shape[1]
                L_gen = text_embedding_delta.shape[1]
                
                # Exact position of gen_text tokens is right after ref_text tokens
                start_pos = ref_len
                if start_pos + L_gen > T_mel:
                    start_pos = max(0, T_mel - L_gen)
                
                avail = min(L_gen, T_mel - start_pos)
                if avail > 0:
                    out_tensor[0:1, start_pos:start_pos + avail, :] += t_delta[:, :avail, :]
                

                if isinstance(output, tuple):
                    return (out_tensor,) + output[1:]
                return out_tensor
                
        hook_handle = target_embed_module.register_forward_hook(hook)

        try:

            ref_duration = speaker_conditioning.get("duration_seconds")
            planner = DurationPlanner()
            planned = planner.plan_duration(
                target_text=text,
                ref_audio_duration=ref_duration,
                ref_text=ref_text,
                language=language,
            )

            # We DO NOT unwrap `infer`. We call it directly so it handles masking and tokenization.
            # CRITICAL: Bypass @torch.no_grad and @torch.inference_mode inside F5-TTS internal sample()
            class _AllowGradContext:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def __call__(self, func): return func

            orig_no_grad = torch.no_grad
            orig_inf_mode = torch.inference_mode

            with torch.enable_grad():
                # Unwrap `infer` to bypass decorator if present
                infer_method = self.tts_api.infer
                if hasattr(infer_method, "__func__"):
                    func = infer_method.__func__
                    if hasattr(func, "__wrapped__"):
                        infer_method = func.__wrapped__.__get__(self.tts_api, self.tts_api.__class__)
                elif hasattr(infer_method, "__wrapped__"):
                    infer_method = infer_method.__wrapped__

                try:
                    torch.no_grad = _AllowGradContext
                    torch.inference_mode = _AllowGradContext
                    try:
                        infer_method(
                            ref_file=temp_ref,
                            ref_text=ref_text,
                            gen_text=text,
                            speed=planned.dynamic_speed_factor,
                            nfe_step=1,  # CRITICAL: Forces 1 outer step.
                            cfg_strength=cfg_strength,
                            target_rms=0.1,
                        )
                    except TypeError:
                        infer_method(
                            ref_file=temp_ref,
                            ref_text=ref_text,
                            gen_text=text,
                            speed=planned.dynamic_speed_factor,
                            nfe_step=1,  # CRITICAL: Forces 1 outer step.
                            cfg_strength=cfg_strength,
                            target_rms=0.1,
                        )
                except (RuntimeError, Exception) as e:
                    if "Can't call numpy() on Tensor that requires grad" in str(e):
                        logger.info("Intercepted final_x with gradients intact (bypassed F5-TTS numpy conversion).")
                    else:
                        raise
                finally:
                    torch.no_grad = orig_no_grad
                    torch.inference_mode = orig_inf_mode

            # The manual vocoding bypass!
            # Because `infer_method` might run under `torch.inference_mode()`, all tensors returned
            # from it will have dropped gradients.
            # But we saved `final_x` directly from `odeint_adjoint` where gradients are intact!
            if not hasattr(wrapper_self, "_saved_final_x"):
                raise RuntimeError("Failed to intercept final_x from ODE solver.")
                
            final_x = wrapper_self._saved_final_x # [B, T_total, 100]
            
            # Transpose to vocoder format [B, 100, T_total]
            pred_mel = final_x.transpose(1, 2)
            
            # Decode manually using the Vocos module components to bypass ANY hidden inference_mode decorators!
            # Vocos.decode essentially does: backbone -> head for mel inputs
            x_vocos = pred_mel.float()
            
            if hasattr(vocoder_obj, "backbone") and hasattr(vocoder_obj, "head"):
                x_vocos = vocoder_obj.backbone(x_vocos)
                waveform = vocoder_obj.head(x_vocos)
            else:
                # Fallback to the unwrapped target_decode
                waveform = target_decode(pred_mel.float())

            # Strip the reference audio part from the generated waveform
            try:
                import soundfile as sf
                ref_data, ref_sr = sf.read(str(temp_ref))
                ref_audio_len = int(len(ref_data) * (24000 / ref_sr))
            except Exception:
                ref_audio_len = 0
                
            if ref_audio_len > 0 and waveform.shape[-1] > ref_audio_len:
                waveform = waveform[..., ref_audio_len:]
                
            logger.warning(f"[DIAG] Final manual waveform requires_grad: {waveform.requires_grad}")

            return waveform, 24000

        finally:
            if hook_handle is not None:
                hook_handle.remove()
            if vocoder_hook_handle is not None:
                vocoder_hook_handle.remove()
            if orig_decode is not None and vocoder_obj is not None:
                vocoder_obj.decode = orig_decode
            if orig_sample is not None:
                model_obj.sample = orig_sample
            dit_model.forward = orig_forward
            torchaudio.load = _orig_load
        

                
    def synthesize_direct(
        self,
        text: str,
        speaker_conditioning: Dict[str, str],
        language: str = "en",
        user_ref_text: Optional[str] = None,
        cfg_strength: float = 2.0,
        text_embedding_delta: Optional[torch.Tensor] = None,
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
        if self.tts_api is None:
            logger.warning("synthesize_direct: F5-TTS is in dummy mode. Returning zero waveform.")
            return torch.zeros(1, 100 * 256), 24000

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

            # --- Text Embedding Hook for Hopfield Corrections ---
            dit_model = getattr(self.model, "transformer", self.model)
            target_embed_module = getattr(dit_model, "text_embed", None)
            hook_handle = None
            if text_embedding_delta is not None and target_embed_module is not None:
                logger.info(f"[Direct] Injecting Hopfield edited text embeddings for inference!")
                gen_tokens = self.get_token_ids(text, language)
                hook_call_count = [0]
                def hook(module, inputs, output):
                    hook_call_count[0] += 1
                    # F5-TTS cfg_infer=True calls text_embed twice sequentially (cond, then uncond).
                    # We MUST only apply the learned memory delta to the conditional stream!
                    if hook_call_count[0] % 2 == 0:
                        return output
                        
                    with torch.set_grad_enabled(False):
                        out_tensor = output.clone()
                        t_delta = text_embedding_delta.to(device=out_tensor.device, dtype=out_tensor.dtype)
                        
                        B = out_tensor.shape[0]
                        T_mel = out_tensor.shape[1]
                        L_gen = text_embedding_delta.shape[1]
                        
                        start_pos = max(0, T_mel - L_gen)
                        
                        # Fuzzy sequence alignment search
                        if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor) and inputs[0].dim() >= 2:
                            tokens = inputs[0][0]
                            sub_seq = gen_tokens[0].to(tokens.device)
                            T_mel_actual = tokens.shape[0]
                            if T_mel_actual >= L_gen:
                                best_match_pos = -1
                                best_match_score = -1
                                for i in range(T_mel_actual - L_gen, -1, -1):
                                    score = (tokens[i:i+L_gen] == sub_seq).sum().item()
                                    if score > best_match_score:
                                        best_match_score = score
                                        best_match_pos = i
                                        
                                if best_match_score >= L_gen * 0.5: # At least 50% match
                                    start_pos = best_match_pos
                                    logger.info(f"[Direct Hook] Fuzzy matched gen_text tokens at pos {start_pos} with score {best_match_score}/{L_gen}")
                                else:
                                    logger.warning(f"[Direct Hook] Could not find a good match for gen_text tokens. Best score: {best_match_score}/{L_gen}. Falling back to default.")

                        avail = min(L_gen, T_mel - start_pos)
                        
                        # Apply to conditional batch ONLY
                        out_tensor[0:1, start_pos:start_pos + avail, :] += t_delta[:, :avail, :]
                        
                        logger.info(f"[Direct Hook] Applied delta. Max delta={t_delta.abs().max().item():.4f}")
                        
                        if isinstance(output, tuple):
                            return (out_tensor,) + output[1:]
                        return out_tensor
                hook_handle = target_embed_module.register_forward_hook(hook)

            # Call F5-TTS inference directly with dynamic speed
            try:
                self.tts_api.infer(
                    ref_file=temp_ref,
                    ref_text=ref_text,
                    gen_text=text,
                    speed=planned.dynamic_speed_factor,
                    nfe_step=32,
                    cfg_strength=cfg_strength,
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
                    cfg_strength=cfg_strength,
                    target_rms=0.1,
                )


            if grabbed_waveforms:
                # Concatenate all batch outputs along the time axis
                waveform = torch.cat(grabbed_waveforms, dim=-1)
                
                # Strip the reference audio part from the generated waveform
                try:
                    import soundfile as sf
                    ref_data, ref_sr = sf.read(str(temp_ref))
                    ref_audio_len = int(len(ref_data) * (24000 / ref_sr))
                except Exception:
                    ref_audio_len = 0
                    
                if ref_audio_len > 0 and waveform.shape[-1] > ref_audio_len:
                    waveform = waveform[..., ref_audio_len:]
                    
                logger.info(f"[Direct] Captured {len(grabbed_waveforms)} batch(es), "
                            f"total waveform length (after stripping ref): {waveform.shape[-1]} samples")
            else:
                raise RuntimeError("Failed to intercept vocoder output tensor.")

            return waveform, 24000

        finally:
            if 'hook_handle' in locals() and hook_handle is not None:
                hook_handle.remove()
            if vocoder_hook_handle is not None:
                vocoder_hook_handle.remove()
            if orig_decode is not None and vocoder_obj is not None:
                vocoder_obj.decode = orig_decode
            torchaudio.load = _orig_load

    def synthesize_baseline(
        self,
        text: str,
        speaker_conditioning: Dict[str, str],
        language: str = "en",
        user_ref_text: Optional[str] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Phase 1: Pure vanilla F5-TTS synthesis WITHOUT hooks and monkey-patches."""
        self._ensure_loaded()
        logger.info("Synthesizing BASELINE via pure F5-TTS (no hooks)...")

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

        try:
            result = self.tts_api.infer(
                ref_file=temp_ref,
                ref_text=ref_text,
                gen_text=text,
                speed=planned.dynamic_speed_factor,
                remove_ref=True,
            )
        except TypeError:
            result = self.tts_api.infer(
                ref_file=temp_ref,
                ref_text=ref_text,
                gen_text=text,
                speed=planned.dynamic_speed_factor,
            )

        import inspect
        import numpy as np
        waveforms = []

        if inspect.isgenerator(result):
            for chunk in result:
                if isinstance(chunk, tuple) and len(chunk) >= 2:
                    wav, sr = chunk[0], chunk[1]
                    waveforms.append(wav)
        else:
            if isinstance(result, tuple) and len(result) >= 2:
                wav, sr = result[0], result[1]
                waveforms.append(wav)

        if not waveforms:
            raise RuntimeError("F5TTS.infer returned no audio chunks in baseline mode.")

        if isinstance(waveforms[0], np.ndarray):
            waveforms = [torch.from_numpy(w).float() for w in waveforms]

        final_wav = torch.cat([w if w.dim() > 1 else w.unsqueeze(0) for w in waveforms], dim=-1)
        if final_wav.dim() == 1:
            final_wav = final_wav.unsqueeze(0)
        elif final_wav.dim() == 3:
            final_wav = final_wav.squeeze(0)

        return final_wav, 24000

    def compute_optimization_loss(
        self,
        text_embedding_delta: torch.Tensor,
        ref_audio_path: str,
        speaker_conditioning: Dict[str, str],
        text: str,
        language: str = "en",
        target_word_start_sample: Optional[int] = None,
        target_word_end_sample: Optional[int] = None,
        seed: Optional[int] = 42,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Stage 2: Compute differentiable Mel-loss for the target word region.

        Paper Section 3.1:
            L_FlowEdit = ||Mel(g_θ(c + δ)) - Mel(y_ref)||_2^2 + λ||δ||_2^2
        """
        import torchaudio
        import torch.nn.functional as F
        from flowedit.utils.audio import AudioProcessor
        
        if target_word_start_sample is None:
            target_word_start_sample = kwargs.get("target_word_start_idx")
        if target_word_end_sample is None:
            target_word_end_sample = kwargs.get("target_word_end_idx")

        # Set reproducible seed for z0 ~ p0 ODE initial condition if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        syn_wav, sr = self.synthesize_from_embeddings(
            text_embedding_delta=text_embedding_delta,
            speaker_conditioning=speaker_conditioning,
            text=text,
            language=language,
            cfg_strength=2.0,
        )
        
        if target_word_start_sample is None or target_word_end_sample is None:
            # Fallback to full waveform if target boundaries not specified
            syn_slice = syn_wav
        else:
            pad = int(0.10 * sr)
            start = max(0, target_word_start_sample - pad)
            end = min(syn_wav.shape[-1], target_word_end_sample + pad)
            syn_slice = syn_wav[..., start:end] if end > start else syn_wav
            
        processor = AudioProcessor()
        syn_mel = processor.compute_mel(syn_slice, normalize=True)
        syn_mel.retain_grad()

        ref_wav, ref_sr = torchaudio.load(ref_audio_path)
        if ref_sr != sr:
            ref_wav = torchaudio.functional.resample(ref_wav, ref_sr, sr)
        ref_wav = ref_wav.to(device=syn_slice.device, dtype=syn_slice.dtype)
        if ref_wav.dim() == 1:
            ref_wav = ref_wav.unsqueeze(0)
            
        ref_mel = processor.compute_mel(ref_wav, normalize=True)

        # Standard log-mel loss computation (Paper Section 3.1)
        if syn_mel.shape[-1] != ref_mel.shape[-1]:
            syn_mel_aligned = F.interpolate(
                syn_mel, 
                size=ref_mel.shape[-1], 
                mode='linear', 
                align_corners=False
            )
        else:
            syn_mel_aligned = syn_mel
            
        loss = F.mse_loss(syn_mel_aligned, ref_mel)
        return {"loss": loss}
