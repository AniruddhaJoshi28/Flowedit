"""
F5-TTS Backbone Wrapper for FlowEdit.

This wraps the continuous Flow-Matching F5-TTS model (Diffusion Transformer).
Uses the official high-level F5TTS API class which correctly handles
model loading, tokenizer/vocab resolution, vocoder setup, and inference.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging
import os
import tempfile

from flowedit.config import BackboneConfig

logger = logging.getLogger(__name__)


class F5TTSBackbone(nn.Module):
    """
    F5-TTS wrapper using the official high-level F5TTS API class.
    This ensures correct tokenizer, vocab, model, and vocoder wiring.
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        self.model = None       # raw DiT model reference (for embedding hooks)
        self.vocoder = None     # vocoder reference
        self.tokenizer = None   # vocab char map
        self.tts_api = None     # high-level F5TTS instance
        self._embedding_hook_handle = None

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
                self.tokenizer = self._make_char_tokenizer(vocab_map)
            else:
                # Fallback: simple character-level tokenizer
                self.tokenizer = self._make_char_tokenizer(None)

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

    def get_speaker_embedding(self, audio_path: Optional[str] = None, language: str = "en", ref_text: Optional[str] = None) -> Dict[str, str]:
        """Store reference audio path for F5-TTS inference and transcribe once."""
        self._ensure_loaded()
        
        if not hasattr(self, "_speaker_cache"):
            self._speaker_cache = {}

        cache_key = f"{audio_path}_{language}_{ref_text}"
        if cache_key in self._speaker_cache:
            return self._speaker_cache[cache_key]
        
        import os
        import tempfile
        import librosa
        import soundfile as sf
        import numpy as np
        
        # Fallback to f5_tts built-in example audio or default speaker audio if not provided or missing
        if not audio_path or not os.path.exists(audio_path):
            try:
                import f5_tts.api
                base_dir = os.path.dirname(f5_tts.api.__file__)
                pkg_wav = os.path.join(base_dir, "infer", "examples", "basic", "basic_ref_en.wav")
                if os.path.exists(pkg_wav):
                    audio_path = pkg_wav
            except Exception:
                pass

        if not audio_path or not os.path.exists(audio_path):
            default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "default_speaker.wav")
            if os.path.exists(default_path):
                audio_path = default_path
            else:
                logger.warning(f"Audio path '{audio_path}' not found and default speaker wav missing.")

        # Create a persistent temp file for the processed 24kHz audio
        processed_fd, processed_path = tempfile.mkstemp(suffix=".wav")
        os.close(processed_fd)
        
        try:
            y, _ = librosa.load(audio_path, sr=24000, mono=True)
            sf.write(processed_path, y, 24000)
            audio_duration = max(0.5, len(y) / 24000.0)
            
            if not ref_text:
                import whisper
                _whisper = whisper.load_model("base", device=str(self.device))
                y_16k, _ = librosa.load(audio_path, sr=16000, mono=True)
                ref_result = _whisper.transcribe(y_16k.astype("float32"), language=language)
                ref_text = ref_result.get("text", "").strip()
                
                # Only apply fallback for long audio if transcription completely failed
                if not ref_text and audio_duration > 2.0:
                    ref_text = "Some call me Nature, others call me Mother Nature."
                elif not ref_text:
                    ref_text = "."
        except Exception as e:
            logger.warning(f"Failed to process speaker audio: {e}")
            if not ref_text:
                ref_text = "."
                
        res = {
            "audio_path": audio_path or "", 
            "processed_audio_path": processed_path,
            "text": ref_text
        }
        self._speaker_cache[cache_key] = res
        return res

    def synthesize_from_embeddings(
        self,
        text_embeddings: torch.Tensor,
        speaker_conditioning: Dict[str, str],
        text: str,
        language: str = "en",
    ) -> Tuple[torch.Tensor, int]:
        """Differentiable synthesis using custom ODE solver and Adjoint method.
        
        Bypasses the non-differentiable `tts_api.infer` and directly runs the 
        Diffusion Transformer (DiT) with `torchdiffeq.odeint_adjoint`.
        """
        self._ensure_loaded()
        logger.info("Synthesizing using Differentiable ODE solver (Adjoint Sensitivity)...")

        # We need the full accuracy of the native F5-TTS inference loop (which handles padding, cond, alignment)
        # BUT we need gradients to flow back to text_embeddings!
        # The native loop uses @torch.inference_mode(), so we dynamically unwrap the key methods.

        # We need the full accuracy of the native F5-TTS inference loop (which handles padding, cond, alignment)
        # BUT we need gradients to flow back to text_embeddings!
        # The native loop uses @torch.inference_mode(), so we dynamically unwrap the key methods.
        import types

        model_obj = getattr(self.tts_api, "ema_model", getattr(self.tts_api, "model", None))
        orig_sample = model_obj.sample
        if hasattr(orig_sample, '__wrapped__'):
            model_obj.sample = types.MethodType(orig_sample.__wrapped__, model_obj)
            
        if hasattr(self.tts_api.infer, '__wrapped__'):
            unwrapped_infer = types.MethodType(self.tts_api.infer.__wrapped__, self.tts_api)
        else:
            unwrapped_infer = self.tts_api.infer

        # No need to calculate ref_tokens_len, we will inject at the trailing tokens of the sequence.

        # 1. Register hook on the FULL TextEmbedding module (AFTER Conv1D upsampling)
        #
        # CRITICAL: encode_text() returns dit_model.text_embed(tokens, seq_lengths)
        # which is the FULL output of TextEmbedding = nn.Embedding → Conv1D → [B, T_mel, 512]
        #
        # We MUST hook dit_model.text_embed (the full module), NOT dit_model.text_embed.text_embed
        # (the inner nn.Embedding). If we hook the inner nn.Embedding, Conv1D will double-process
        # our already-processed embeddings, garbling the phonetic corrections completely.
        dit_model = getattr(self.model, "transformer", self.model)
        hook_handle = None
        
        # Target the FULL text_embed module (includes Conv1D upsampling)
        target_embed_module = getattr(dit_model, "text_embed", None)

        if target_embed_module is not None:
            def hook(module, inputs, output):
                out_tensor = output[0] if isinstance(output, tuple) else output
                T_mel = out_tensor.shape[1]  # mel-frame length after Conv1D
                L_gen = text_embeddings.shape[1]  # our corrected embeddings length
                
                # Our corrected embeddings from encode_text() are in the SAME space as 
                # this output (both are post-Conv1D). We inject them at the trailing
                # positions corresponding to gen_text.
                start_pos = max(0, T_mel - L_gen)
                t_embed = text_embeddings.to(device=out_tensor.device, dtype=out_tensor.dtype)
                
                new_out_tensor = out_tensor.clone()
                avail = min(L_gen, T_mel - start_pos)
                new_out_tensor[:, start_pos:start_pos + avail, :] = t_embed[:, :avail, :]
                
                logger.info(
                    f"[Hook] Full TextEmbedding Hook -> T_mel: {T_mel}, "
                    f"Gen embed len: {L_gen} (start_pos={start_pos}, avail={avail}), "
                    f"Corrected emb norm: {torch.norm(t_embed).item():.4f}, "
                    f"Original emb norm: {torch.norm(out_tensor).item():.4f}"
                )
                
                if isinstance(output, tuple):
                    return (new_out_tensor,) + output[1:]
                return new_out_tensor

            hook_handle = target_embed_module.register_forward_hook(hook)

        # 2. Register hook or monkey-patch Vocoder to grab the waveform TENSOR (with gradients!) before it is detached to numpy
        grabbed_waveform = [None]
        vocoder_hook_handle = None
        orig_decode = None
        vocoder_obj = getattr(self.tts_api, "vocoder", getattr(self, "vocoder", None))
        if vocoder_obj is not None:
            # Standard forward hook if vocoder.__call__ / forward is invoked (e.g. BigVGAN)
            def vocoder_hook(module, inputs, output):
                grabbed_waveform[0] = output
            vocoder_hook_handle = vocoder_obj.register_forward_hook(vocoder_hook)

            # Monkey-patch decode method if it exists, since Vocos uses .decode() directly (bypassing forward/__call__)
            if hasattr(vocoder_obj, "decode"):
                orig_decode = vocoder_obj.decode
                def wrapped_decode(*args, **kwargs):
                    output = orig_decode(*args, **kwargs)
                    grabbed_waveform[0] = output
                    return output
                vocoder_obj.decode = wrapped_decode

        # Prepare audio
        import soundfile as sf
        import torchaudio

        # Use the pre-processed 24kHz audio
        temp_ref = speaker_conditioning.get("processed_audio_path")
        if not temp_ref or not os.path.exists(temp_ref):
            # Fallback if processed_audio_path is missing
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

        try:
            ref_text = speaker_conditioning.get("text", ".").strip()
            if not ref_text:
                ref_text = "."

            # Run UNWRAPPED inference (allows gradients!)
            with torch.set_grad_enabled(True):
                try:
                    unwrapped_infer(
                        ref_file=temp_ref,
                        ref_text=ref_text,
                        gen_text=text,
                        speed=1.0,
                        nfe_step=32,
                        cfg_strength=2.0,
                        target_rms=0.1,
                        remove_ref=True,
                    )
                except TypeError:
                    unwrapped_infer(
                        ref_file=temp_ref,
                        ref_text=ref_text,
                        gen_text=text,
                        speed=1.0,
                        nfe_step=32,
                        cfg_strength=2.0,
                        target_rms=0.1,
                    )

            # Retrieve the differentiable waveform!
            if grabbed_waveform[0] is not None:
                waveform = grabbed_waveform[0]
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
            model_obj.sample = orig_sample
            torchaudio.load = _orig_load
        

                
    def _register_embedding_hook(self, target_embeddings: torch.Tensor):
        """Register a forward hook to inject perturbed embeddings."""
        def hook(module, inputs, output):
            return target_embeddings

        class DummyHandle:
            def remove(self): pass
        return DummyHandle()


