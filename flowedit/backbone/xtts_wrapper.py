"""
XTTS-v2 Backbone Wrapper for FlowEdit.

This wraps the autoregressive XTTS-v2 model (GPT-based) from Coqui TTS.
Unlike F5-TTS (flow-matching DiT), XTTS-v2 uses:
  - GPT-2 style autoregressive decoder for audio token prediction
  - VQ-VAE decoder (HiFi-GAN) for waveform generation
  - Speaker conditioning via latent embeddings extracted from reference audio

The model files are loaded from a local directory (flowedit/Xtts/) containing:
  - model.pth       : Fine-tuned GPT checkpoint
  - config.json     : Model configuration
  - vocab.json      : BPE tokenizer vocabulary
  - dvae.pth        : Discrete VAE checkpoint
  - mel_norms.pth   : Mel-spectrogram normalization stats
  - mel_stats.pth   : Mel statistics

Reference: Adapted for FlowEdit (arXiv:2606.20518) backbone interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
import tempfile
import json

from flowedit.config import BackboneConfig
from flowedit.backbone.base import TTSBackbone

logger = logging.getLogger(__name__)


class XTTSBackbone(TTSBackbone):
    """
    XTTS-v2 wrapper using the Coqui TTS library.

    Provides the same interface as F5TTSBackbone so the FlowEdit pipeline
    can use either backbone interchangeably via the factory function.
    """

    def __init__(self, config: BackboneConfig):
        super().__init__(config)
        self.device_name = config.device
        self.model = None          # Xtts model instance
        self.tokenizer_instance = None      # BPE tokenizer
        self._xtts_config = None   # XttsConfig instance
        self._embedding_dim = None
        self._speaker_cache = {}

    @property
    def device(self) -> str:
        return self.device_name

    @property
    def tokenizer(self):
        return self.tokenizer_instance

    @property
    def optimization_mode(self) -> str:
        return "teacher_forcing"

    @property
    def embedding_dim(self) -> int:
        """Return the dimension of the text/GPT embeddings."""
        if self._embedding_dim is not None:
            return self._embedding_dim
        if self.model is not None:
            # XTTS GPT model channel dimension
            try:
                gpt = getattr(self.model, "gpt", None)
                if gpt is not None:
                    # gpt.n_model_channels or from config
                    dim = getattr(gpt, "n_model_channels", None)
                    if dim:
                        self._embedding_dim = dim
                        return dim
            except Exception:
                pass
            # Fallback: read from config
            try:
                self._embedding_dim = self._xtts_config.model_args.get(
                    "gpt_n_model_channels", 1024
                )
                return self._embedding_dim
            except Exception:
                pass
        return 1024  # XTTS-v2 default

    def _resolve_model_dir(self) -> str:
        """Resolve the XTTS model directory path."""
        candidates = []

        model_dir = getattr(self.config, "xtts_model_dir", None)
        if model_dir:
            candidates.append(model_dir)

        # 1. Package directory: flowedit/Xtts
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates.append(os.path.join(pkg_dir, "Xtts"))

        # 2. Parent directory: flowedit/../Xtts
        repo_root = os.path.dirname(pkg_dir)
        candidates.append(os.path.join(repo_root, "Xtts"))
        candidates.append(os.path.join(repo_root, "flowedit", "Xtts"))
        candidates.append(os.path.join(repo_root, "Flowedit", "Xtts"))
        candidates.append(os.path.join(repo_root, "Flowedit", "flowedit", "Xtts"))

        # Return the first candidate that exists and contains vocab.json
        for candidate in candidates:
            if candidate and os.path.isdir(candidate):
                vocab_check = os.path.join(candidate, "vocab.json")
                if os.path.isfile(vocab_check):
                    return candidate

        # Fallback: if candidate directory exists even without vocab.json
        for candidate in candidates:
            if candidate and os.path.isdir(candidate):
                return candidate

        checked_paths = "\n  ".join(f"- {c}" for c in candidates if c)
        raise FileNotFoundError(
            f"XTTS model directory not found or missing vocab.json. Checked:\n  {checked_paths}\n"
            f"Please set config.backbone.xtts_model_dir to the correct path."
        )

    def load_model(self):
        """Load the XTTS-v2 model from local files."""
        model_dir = self._resolve_model_dir()
        checkpoint = getattr(self.config, "xtts_checkpoint", "model.pth")
        checkpoint_path = os.path.join(model_dir, checkpoint)
        config_path = os.path.join(model_dir, "config.json")
        vocab_path = os.path.join(model_dir, "vocab.json")

        logger.info(f"Loading XTTS-v2 from local files: {model_dir}")
        logger.info(f"  Checkpoint: {checkpoint}")
        logger.info(f"  Config: config.json")
        logger.info(f"  Vocab: vocab.json")

        # Validate required files exist
        for fpath, fname in [
            (checkpoint_path, checkpoint),
            (config_path, "config.json"),
            (vocab_path, "vocab.json"),
        ]:
            if not os.path.isfile(fpath):
                raise FileNotFoundError(f"Required XTTS file not found: {fpath}")

        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
        except ImportError:
            logger.error(
                "Coqui TTS is not installed! Run: pip install coqui-tts\n"
                "Note: The original 'TTS' package is deprecated and doesn't "
                "support Python 3.12+. Use 'coqui-tts' (same API)."
            )
            raise

        # Verify Checkpoint Identity (Phase 0.5)
        import hashlib
        try:
            with open(checkpoint_path, "rb") as f:
                file_hash = hashlib.sha256()
                while chunk := f.read(8192 * 1024):  # 8MB chunks
                    file_hash.update(chunk)
            sha256_hash = file_hash.hexdigest()
        except Exception as e:
            sha256_hash = f"Error computing hash: {e}"

        logger.info("=" * 60)
        logger.info("XTTS CHECKPOINT VERIFICATION (Phase 0.5)")
        logger.info(f"  Path: {checkpoint_path}")
        logger.info(f"  SHA256: {sha256_hash}")
        logger.info(f"  Model Name: XTTS-v2")
        logger.info("=" * 60)

        # Load XTTS config
        xtts_config = XttsConfig()
        xtts_config.load_json(config_path)
        self._xtts_config = xtts_config

        # Initialize model
        self.model = Xtts.init_from_config(xtts_config)
        try:
            self.model.load_checkpoint(
                xtts_config,
                checkpoint_dir=model_dir,
                checkpoint_path=checkpoint_path,
                vocab_path=vocab_path,
                eval=True,
                use_deepspeed=False,
                strict=False,
            )
        except KeyError as e:
            if "model" in str(e) or "'model'" in str(e):
                logger.warning("Checkpoint missing 'model' key. Wrapping state dict in a temporary file...")
                raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                wrapped_checkpoint = {"model": raw_checkpoint.get("state_dict", raw_checkpoint)}
                
                fd, tmp_path = tempfile.mkstemp(suffix=".pth")
                os.close(fd)
                try:
                    torch.save(wrapped_checkpoint, tmp_path)
                    self.model.load_checkpoint(
                        xtts_config,
                        checkpoint_dir=model_dir,
                        checkpoint_path=tmp_path,
                        vocab_path=vocab_path,
                        eval=True,
                        use_deepspeed=False,
                        strict=False,
                    )
                finally:
                    os.remove(tmp_path)
            else:
                raise
        self.model = self.model.to(self.device)

        # Ensure DVAE is loaded (needed for compute_optimization_loss)
        if not hasattr(self.model, "dvae") or self.model.dvae is None:
            try:
                from TTS.tts.layers.xtts.dvae import DiscreteVAE
                dvae = DiscreteVAE(
                    channels=80,
                    normalization=None,
                    positional_dims=1,
                    num_tokens=1024,
                    codebook_dim=512,
                    hidden_dim=512,
                    num_resnet_blocks=3,
                    kernel_size=3,
                    num_layers=2,
                    use_transposed_convs=False,
                )
                dvae_path = os.path.join(model_dir, "dvae.pth")
                if os.path.exists(dvae_path):
                    dvae_checkpoint = torch.load(dvae_path, map_location="cpu", weights_only=False)
                    if "model" in dvae_checkpoint:
                        dvae.load_state_dict(dvae_checkpoint["model"], strict=False)
                    else:
                        dvae.load_state_dict(dvae_checkpoint, strict=False)
                    dvae = dvae.to(self.device)
                    dvae.eval()
                    self.model.dvae = dvae
                    logger.info("XTTS DVAE loaded successfully.")
                else:
                    logger.warning(f"XTTS DVAE checkpoint not found at {dvae_path}. Loss computation will fail if attempted.")
            except Exception as e:
                logger.error(f"Failed to load XTTS DVAE: {e}")

        # Extract tokenizer reference
        self.tokenizer_instance = getattr(self.model, "tokenizer", None)

        # Cache embedding dim
        _ = self.embedding_dim

        logger.info(
            f"XTTS-v2 loaded successfully. "
            f"Device: {self.device}, Embedding dim: {self.embedding_dim}"
        )

    def _ensure_loaded(self):
        if self.model is None:
            self.load_model()

    def get_token_ids(self, text: str, language: str = "en") -> torch.Tensor:
        """Tokenize text into IDs using the XTTS BPE tokenizer."""
        self._ensure_loaded()

        if self.tokenizer is not None:
            try:
                # Coqui XTTS tokenizer.encode returns list of token IDs
                token_ids = self.tokenizer.encode(text, lang=language)
                if isinstance(token_ids, list):
                    if isinstance(token_ids[0], list):
                        token_ids = token_ids[0]
                return torch.tensor(
                    token_ids, dtype=torch.long, device=self.device
                ).unsqueeze(0)
            except Exception as e:
                logger.warning(f"Tokenizer encode failed: {e}, using fallback")

        # Fallback: character-level
        tokens = torch.zeros(
            (1, len(text)), dtype=torch.long, device=self.device
        )
        return tokens

    def encode_text(self, text: str, language: str = "en") -> torch.Tensor:
        """
        Encode text into continuous embeddings.

        For XTTS, this extracts GPT text embeddings by running the text
        through the GPT model's text embedding layer.

        Used by Hopfield Memory for key computation.
        """
        self._ensure_loaded()
        tokens = self.get_token_ids(text, language)

        with torch.no_grad():
            gpt = getattr(self.model, "gpt", None)
            if gpt is not None:
                # Try to access the text embedding layer of the GPT model
                text_embed = getattr(gpt, "text_embedding", None)
                if text_embed is None:
                    text_embed = getattr(gpt, "text_embed", None)

                if text_embed is not None:
                    try:
                        embeddings = text_embed(tokens)
                        if isinstance(embeddings, tuple):
                            embeddings = embeddings[0]
                        return embeddings
                    except Exception as e:
                        logger.warning(
                            f"GPT text_embedding failed: {e}, trying mel_embedding"
                        )

                # Alternative: use the text_head or conditioning encoder
                cond_encoder = getattr(gpt, "text_pos_embedding", None)
                if cond_encoder is not None:
                    try:
                        embeddings = cond_encoder(tokens)
                        if isinstance(embeddings, tuple):
                            embeddings = embeddings[0]
                        return embeddings
                    except Exception:
                        pass

            # Fallback: use a simple embedding table
            if not hasattr(self, "_fallback_embed"):
                self._fallback_embed = nn.Embedding(
                    10000, self.embedding_dim
                ).to(self.device)
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
        """Compute differentiable teacher-forced CE loss for XTTS latent optimization.

        Creates a direct gradient path: δ → perturbed_embeddings → CE loss over target audio tokens.
        """
        self._ensure_loaded()
        gpt = getattr(self.model, "gpt", None)
        if gpt is None:
            raise RuntimeError("XTTS model does not have a GPT module loaded.")

        import torchaudio
        device = self.device

        # 1. Prepare inputs (audio_codes, tokens, etc.)
        tokens = self.get_token_ids(text, language)
        text_lengths = torch.tensor([tokens.shape[1]], dtype=torch.long, device=device)

        # 2. Extract reference audio features (teacher targets)
        waveform, sr = torchaudio.load(ref_audio_path)
        waveform = waveform.to(device)
        if sr != 22050:
            import torchaudio.transforms as T
            resampler = T.Resample(sr, 22050).to(device)
            waveform = resampler(waveform)

        if not hasattr(self.model, 'mel_transform'):
            self.model.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=22050, n_fft=1024, win_length=1024, hop_length=256, f_min=0, f_max=8000, n_mels=80
            ).to(device)

        mel = self.model.mel_transform(waveform)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        with torch.no_grad():
            audio_codes = self.model.dvae.get_codebook_indices(mel)
        audio_lengths = torch.tensor([audio_codes.shape[1]], dtype=torch.long, device=device)

        gpt_cond_latent = speaker_conditioning.get("gpt_cond_latent")
        if gpt_cond_latent is None:
            raise RuntimeError("Missing gpt_cond_latent in speaker_conditioning.")

        # 3. Hook the embedding layer to inject perturbed_embeddings
        target_embed_layer = getattr(gpt, "text_embedding", getattr(gpt, "text_embed", None))
        if target_embed_layer is None:
            raise RuntimeError("Cannot find text_embedding layer in GPT")

        hook_handle = None

        def embedding_hook(module, inputs, output):
            out_tensor = output[0] if isinstance(output, tuple) else output
            new_out = out_tensor.clone()
            t_embed = perturbed_embeddings.to(
                device=out_tensor.device, dtype=out_tensor.dtype
            )

            input_ids = inputs[0]  # Expected shape [batch, T_seq]
            L_corr = t_embed.shape[1]
            T_seq = out_tensor.shape[1]

            start_pos = -1
            if input_ids is not None and input_ids.dim() >= 2 and tokens.shape[1] <= input_ids.shape[1]:
                # Search for the exact token subsequence
                my_seq = tokens[0].to(input_ids.device)
                for i in range(input_ids.shape[1] - my_seq.shape[0] + 1):
                    if torch.all(input_ids[0, i:i+my_seq.shape[0]] == my_seq):
                        start_pos = i
                        break

            if start_pos == -1:
                start_pos = max(0, T_seq - L_corr)

            avail = min(L_corr, T_seq - start_pos)
            new_out[:, start_pos : start_pos + avail, :] = t_embed[
                :, :avail, :
            ]

            if isinstance(output, tuple):
                return (new_out,) + output[1:]
            return new_out

        hook_handle = target_embed_layer.register_forward_hook(embedding_hook)

        try:
            # 4. Forward pass through GPT to compute loss
            res = gpt.forward(
                text_inputs=tokens,
                text_lengths=text_lengths,
                audio_codes=audio_codes,
                wav_lengths=audio_lengths,
                cond_mels=mel,
                cond_latents=gpt_cond_latent,
            )

            if isinstance(res, tuple):
                loss_text = res[0]
                loss_mel = res[1]
                loss = loss_text + loss_mel
            elif isinstance(res, dict):
                loss = res.get("loss", sum(res.values()))
                loss_text = res.get("loss_text", loss)
            else:
                loss = res
                loss_text = loss

            return {
                "loss": loss,
                "ce_loss": loss_text.item() if hasattr(loss_text, "item") else float(loss_text),
                "embedding_norm": torch.norm(perturbed_embeddings).item(),
            }
        finally:
            if hook_handle is not None:
                hook_handle.remove()


    def get_speaker_embedding(
        self,
        audio_path: Optional[str] = None,
        language: str = "en",
        ref_text: Optional[str] = None,
    ) -> Dict:
        """
        Extract speaker conditioning from reference audio.

        XTTS uses two forms of speaker conditioning:
          - gpt_cond_latent: GPT conditioning latent from reference audio
          - speaker_embedding: Speaker embedding vector

        Returns a dict with these tensors plus metadata.
        """
        self._ensure_loaded()

        cache_key = f"{audio_path}_{language}_{ref_text}"
        if cache_key in self._speaker_cache:
            return self._speaker_cache[cache_key]

        import librosa
        import soundfile as sf

        # Resolve audio path fallbacks
        if not audio_path or not os.path.exists(audio_path):
            default_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "resources",
                "default_speaker.wav",
            )
            if os.path.exists(default_path):
                audio_path = default_path
            else:
                logger.warning(
                    f"Audio path '{audio_path}' not found and no default speaker."
                )

        # Create processed temp file at 22050 Hz (XTTS input sample rate)
        processed_fd, processed_path = tempfile.mkstemp(suffix=".wav")
        os.close(processed_fd)

        try:
            import soundfile as sf
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            sf.write(processed_path, y, 22050, subtype='PCM_16')

            # Extract XTTS conditioning latents using the clean processed WAV
            gpt_cond_latent, speaker_embedding = (
                self.model.get_conditioning_latents(
                    audio_path=[processed_path],
                    gpt_cond_len=self._get_config_value("gpt_cond_len", 12),
                    gpt_cond_chunk_len=self._get_config_value(
                        "gpt_cond_chunk_len", 4
                    ),
                    max_ref_length=self._get_config_value("max_ref_len", 10),
                )
            )

            # Auto-transcribe if no ref_text provided
            if not ref_text:
                try:
                    import whisper

                    _whisper = whisper.load_model(
                        "base", device=str(self.device)
                    )
                    y_16k, _ = librosa.load(processed_path, sr=16000, mono=True)
                    ref_result = _whisper.transcribe(
                        y_16k.astype("float32"), language=language
                    )
                    ref_text = ref_result.get("text", "").strip()
                except Exception as e:
                    logger.warning(f"Whisper transcription failed: {e}")

                if not ref_text:
                    ref_text = "."

        except Exception as e:
            logger.error(f"Failed to process speaker audio: {e}", exc_info=True)
            gpt_cond_latent = None
            speaker_embedding = None
            if not ref_text:
                ref_text = "."

        result = {
            "audio_path": audio_path or "",
            "processed_audio_path": processed_path,
            "text": ref_text,
            "gpt_cond_latent": gpt_cond_latent,
            "speaker_embedding": speaker_embedding,
        }
        self._speaker_cache[cache_key] = result
        return result

    def synthesize_from_embeddings(
        self,
        text_embeddings: torch.Tensor,
        speaker_conditioning: Dict,
        text: str,
        language: str = "en",
    ) -> Tuple[torch.Tensor, int]:
        """
        Synthesis with modified text embeddings (for Hopfield corrections).

        For XTTS, we hook into the GPT text embedding layer to inject
        the corrected embeddings, similar to the F5-TTS approach.
        """
        self._ensure_loaded()
        logger.info(
            "Synthesizing with embedding injection (XTTS hook-based)..."
        )

        gpt_cond_latent = speaker_conditioning.get("gpt_cond_latent")
        speaker_emb = speaker_conditioning.get("speaker_embedding")

        if gpt_cond_latent is None or speaker_emb is None:
            logger.warning(
                "Missing XTTS conditioning latents, falling back to direct synthesis"
            )
            return self.synthesize_direct(
                text, speaker_conditioning, language
            )

        # Register hook on GPT text embedding to inject corrections
        gpt = getattr(self.model, "gpt", None)
        hook_handle = None

        if gpt is not None:
            target_embed = getattr(
                gpt, "text_embedding", getattr(gpt, "text_embed", None)
            )
            if target_embed is not None:
                # Precompute the token sequence we are injecting
                my_tokens = self.get_token_ids(text, language)

                def hook(module, inputs, output):
                    out_tensor = (
                        output[0] if isinstance(output, tuple) else output
                    )
                    new_out = out_tensor.clone()
                    t_embed = text_embeddings.to(
                        device=out_tensor.device, dtype=out_tensor.dtype
                    )

                    input_ids = inputs[0]  # Expected shape [batch, T_seq]
                    L_corr = t_embed.shape[1]
                    T_seq = out_tensor.shape[1]

                    start_pos = -1
                    if input_ids is not None and input_ids.dim() >= 2 and my_tokens.shape[1] <= input_ids.shape[1]:
                        # Search for the exact token subsequence
                        my_seq = my_tokens[0].to(input_ids.device)
                        for i in range(input_ids.shape[1] - my_seq.shape[0] + 1):
                            if torch.all(input_ids[0, i:i+my_seq.shape[0]] == my_seq):
                                start_pos = i
                                break

                    if start_pos == -1:
                        print("[XTTS Hook] Exact token sequence not found in inputs. Falling back to right-align.")
                        start_pos = max(0, T_seq - L_corr)

                    avail = min(L_corr, T_seq - start_pos)
                    new_out[:, start_pos : start_pos + avail, :] = t_embed[
                        :, :avail, :
                    ]

                    print(
                        f"[XTTS Hook] T_seq: {T_seq}, Corr len: {L_corr}, "
                        f"start_pos: {start_pos}, avail: {avail}"
                    )

                    if isinstance(output, tuple):
                        return (new_out,) + output[1:]
                    return new_out

                hook_handle = target_embed.register_forward_hook(hook)
                print(f"[XTTS] Successfully registered hook on {target_embed.__class__.__name__}")
            else:
                print("[XTTS ERROR] Could not find text_embedding in GPT!")

        try:
            # Run XTTS inference
            out = self.model.inference(
                text=text,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_emb,
                temperature=0.75,
                length_penalty=1.0,
                repetition_penalty=10.0,
                top_k=50,
                top_p=0.85,
                enable_text_splitting=False,
            )

            waveform = out.get("wav", None)
            if waveform is None:
                raise RuntimeError("XTTS inference returned no waveform")

            if isinstance(waveform, torch.Tensor):
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
            else:
                import numpy as np

                waveform = torch.from_numpy(np.array(waveform)).float()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)

            sample_rate = self._get_config_value("output_sample_rate", 24000)
            return waveform, sample_rate

        finally:
            if hook_handle is not None:
                hook_handle.remove()

    def synthesize_direct(
        self,
        text: str,
        speaker_conditioning: Dict,
        language: str = "en",
        user_ref_text: Optional[str] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Direct XTTS synthesis WITHOUT embedding injection hooks.

        Used when no Hopfield corrections are active (or corrections don't
        change the embeddings).
        """
        self._ensure_loaded()
        logger.info("Synthesizing directly via XTTS (no embedding hooks)...")

        gpt_cond_latent = speaker_conditioning.get("gpt_cond_latent")
        speaker_emb = speaker_conditioning.get("speaker_embedding")

        if gpt_cond_latent is None or speaker_emb is None:
            raise RuntimeError(
                "Missing XTTS conditioning latents. "
                "Ensure get_speaker_embedding() was called with valid audio."
            )

        out = self.model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_emb,
            temperature=0.75,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=50,
            top_p=0.85,
            enable_text_splitting=False,
        )

        waveform = out.get("wav", None)
        if waveform is None:
            raise RuntimeError("XTTS inference returned no waveform")

        if isinstance(waveform, torch.Tensor):
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
        else:
            import numpy as np

            waveform = torch.from_numpy(np.array(waveform)).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

        sample_rate = self._get_config_value("output_sample_rate", 24000)
        return waveform, sample_rate

    def _get_config_value(self, key: str, default):
        """Read a value from the XTTS config JSON, with a fallback default."""
        if self._xtts_config is not None:
            # Try top-level config attributes
            val = getattr(self._xtts_config, key, None)
            if val is not None:
                return val
            # Try model_args
            model_args = getattr(self._xtts_config, "model_args", None)
            if model_args is not None:
                if isinstance(model_args, dict):
                    val = model_args.get(key, None)
                else:
                    val = getattr(model_args, key, None)
                if val is not None:
                    return val
        return default
