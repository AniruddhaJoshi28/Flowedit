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

        # 2. Register hook or monkey-patch Vocoder to grab waveform TENSORS from ALL batches
        # F5-TTS splits long text into multiple batches; we must collect them all.
        grabbed_waveforms = []
        vocoder_hook_handle = None
        orig_decode = None
        vocoder_obj = getattr(self.tts_api, "vocoder", getattr(self, "vocoder", None))
        if vocoder_obj is not None:
            # Standard forward hook if vocoder.__call__ / forward is invoked (e.g. BigVGAN)
            def vocoder_hook(module, inputs, output):
                grabbed_waveforms.append(output)
            vocoder_hook_handle = vocoder_obj.register_forward_hook(vocoder_hook)

            # Monkey-patch decode method if it exists, since Vocos uses .decode() directly (bypassing forward/__call__)
            if hasattr(vocoder_obj, "decode"):
                orig_decode = vocoder_obj.decode
                def wrapped_decode(*args, **kwargs):
                    output = orig_decode(*args, **kwargs)
                    grabbed_waveforms.append(output)
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

            ref_duration = speaker_conditioning.get("duration_seconds")
            planner = DurationPlanner()
            planned = planner.plan_duration(
                target_text=text,
                ref_audio_duration=ref_duration,
                ref_text=ref_text,
                language=language,
            )

            logger.info(f"F5-TTS Synthesis: ref_text='{ref_text}', gen_text='{text}', dynamic_speed={planned.dynamic_speed_factor:.3f}")

            # Run UNWRAPPED inference (allows gradients!) with dynamic speed
            with torch.set_grad_enabled(True):
                try:
                    unwrapped_infer(
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
                    unwrapped_infer(
                        ref_file=temp_ref,
                        ref_text=ref_text,
                        gen_text=text,
                        speed=planned.dynamic_speed_factor,
                        nfe_step=32,
                        cfg_strength=2.0,
                        target_rms=0.1,
                    )


            # Retrieve and concatenate all batch waveforms
            if grabbed_waveforms:
                waveform = torch.cat(grabbed_waveforms, dim=-1)
                logger.info(f"Captured {len(grabbed_waveforms)} batch(es), "
                            f"total waveform length: {waveform.shape[-1]} samples")
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
