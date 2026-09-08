"""
XTTS-v2 Backbone Wrapper for FlowEdit.

Implements the TTSBackbone interface for the fine-tuned XTTS-v2 model.
Loads checkpoints, configuration, tokenizer vocab, and Discrete VAE from
the fine-tuned text_to_speech/app/model directory.

Paper Reference: FlowEdit (arXiv:2606.20518), Section 3.1 & 3.2.
"""

import os
import json
import logging
import tempfile
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import librosa
except ImportError:
    librosa = None

try:
    import soundfile as sf
except ImportError:
    sf = None

from flowedit.config import BackboneConfig
from flowedit.backbone.base import TTSBackbone

logger = logging.getLogger(__name__)


class XTTSBackbone(TTSBackbone):
    """
    XTTS-v2 backbone wrapper supporting fine-tuned model checkpoints.
    Interacts seamlessly with the FlowEdit pipeline via the TTSBackbone contract.
    """

    def __init__(self, config: Optional[BackboneConfig] = None):
        if config is None:
            config = BackboneConfig()
        super().__init__(config)
        self.device_name = getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer_instance = None
        self._xtts_config = None
        self._embedding_dim = 1024
        self._speaker_cache = {}

    @property
    def device(self) -> str:
        return self.device_name

    @property
    def tokenizer(self):
        return self.tokenizer_instance

    @property
    def embedding_dim(self) -> int:
        """Return text/GPT embedding channel dimension d (Paper Section 3.1: d = 1024)."""
        if self._embedding_dim is not None:
            return self._embedding_dim
        if self.model is not None:
            try:
                gpt = getattr(self.model, "gpt", None)
                if gpt is not None:
                    dim = getattr(gpt, "n_model_channels", None)
                    if dim:
                        self._embedding_dim = dim
                        return dim
            except Exception:
                pass
            try:
                if self._xtts_config is not None:
                    model_args = getattr(self._xtts_config, "model_args", {})
                    if isinstance(model_args, dict):
                        dim = model_args.get("gpt_n_model_channels", 1024)
                    else:
                        dim = getattr(model_args, "gpt_n_model_channels", 1024)
                    self._embedding_dim = dim
                    return dim
            except Exception:
                pass
        return 1024

    def _resolve_model_dir(self) -> str:
        """Resolve the directory containing fine-tuned XTTS model files."""
        candidates = []

        # 1. Configured directory override
        model_dir = getattr(self.config, "xtts_model_dir", "")
        if model_dir:
            candidates.append(model_dir)

        # 2. Environment variables
        for env_var in ["FLOWEDIT_XTTS_DIR", "FLOWEDIT_MODEL_DIR", "MODEL_DIR"]:
            env_val = os.environ.get(env_var, "")
            if env_val:
                candidates.append(env_val)

        # 3. Path relative to this file, workspace, or sibling folders
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        flowedit_root = os.path.dirname(pkg_dir)
        workspace_root = os.path.dirname(flowedit_root)

        candidates.append(os.path.join(workspace_root, "text_to_speech", "app", "model"))
        candidates.append(os.path.join(workspace_root, "..", "text_to_speech", "app", "model"))
        candidates.append(os.path.join(flowedit_root, "..", "text_to_speech", "app", "model"))
        candidates.append(os.path.join(flowedit_root, "text_to_speech", "app", "model"))
        candidates.append(os.path.join(os.getcwd(), "text_to_speech", "app", "model"))
        candidates.append(os.path.join(os.getcwd(), "..", "text_to_speech", "app", "model"))
        candidates.append(os.path.join(flowedit_root, "models", "xtts"))
        candidates.append(os.path.join(workspace_root, "models", "xtts"))
        candidates.append(os.path.join(flowedit_root, "flowedit", "Xtts"))
        candidates.append(os.path.join(flowedit_root, "Xtts"))
        candidates.append(os.path.join(pkg_dir, "Xtts"))

        # User home folder project conventions (e.g. Linux / Windows servers)
        try:
            home = os.path.expanduser("~")
            candidates.append(os.path.join(home, "projects", "text_to_speech", "app", "model"))
            candidates.append(os.path.join(home, "projects", "flow_edit", "text_to_speech", "app", "model"))
        except Exception:
            pass

        # Standard hardcoded fallback for server environment
        candidates.append("/home/rsurya/projects/text_to_speech/app/model")

        # Find the first existing candidate directory containing vocab.json or config.json
        for candidate in candidates:
            if candidate and os.path.isdir(candidate):
                vocab_check = os.path.join(candidate, "vocab.json")
                config_check = os.path.join(candidate, "config.json")
                if os.path.isfile(vocab_check) or os.path.isfile(config_check):
                    logger.info(f"Resolved XTTS model directory: {candidate}")
                    return os.path.abspath(candidate)

        for candidate in candidates:
            if candidate and os.path.isdir(candidate):
                return os.path.abspath(candidate)

        checked_paths = "\n  ".join(f"- {c}" for c in candidates if c)
        raise FileNotFoundError(
            f"Fine-tuned XTTS model directory not found. Checked paths:\n  {checked_paths}\n"
            "Please ensure text_to_speech/app/model is present or set FLOWEDIT_XTTS_DIR or config.backbone.xtts_model_dir."
        )

    def load_model(self) -> None:
        """Load fine-tuned XTTS model, BPE tokenizer, and Discrete VAE."""
        model_dir = self._resolve_model_dir()
        checkpoint_name = getattr(self.config, "xtts_checkpoint", "model.pth") or "model.pth"
        config_name = getattr(self.config, "xtts_config_file", "config.json") or "config.json"
        vocab_name = getattr(self.config, "xtts_vocab_file", "vocab.json") or "vocab.json"
        dvae_name = getattr(self.config, "xtts_dvae_file", "dvae.pth") or "dvae.pth"

        checkpoint_path = os.path.join(model_dir, checkpoint_name)
        config_path = os.path.join(model_dir, config_name)
        vocab_path = os.path.join(model_dir, vocab_name)
        dvae_path = os.path.join(model_dir, dvae_name)

        logger.info(f"Loading Fine-Tuned XTTS Backbone on device '{self.device}'...")
        logger.info(f"  Model Dir:   {model_dir}")
        logger.info(f"  Checkpoint:  {checkpoint_path}")
        logger.info(f"  Config:      {config_path}")
        logger.info(f"  Vocab:       {vocab_path}")

        # Validate required files exist
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"XTTS Checkpoint not found at: {checkpoint_path}")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"XTTS Config not found at: {config_path}")
        if not os.path.isfile(vocab_path):
            raise FileNotFoundError(f"XTTS Vocab not found at: {vocab_path}")

        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
            from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
        except ImportError:
            logger.error(
                "Coqui TTS is required for XTTS backbone. Please install via 'pip install coqui-tts' or 'pip install TTS'."
            )
            raise

        # Load XTTS configuration
        xtts_config = XttsConfig()
        xtts_config.load_json(config_path)
        self._xtts_config = xtts_config

        # Initialize XTTS model
        self.model = Xtts.init_from_config(xtts_config)

        # Load weights with compatibility fallback for wrapped/raw state dicts
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
                logger.warning("Checkpoint missing 'model' key. Wrapping state dict in temp checkpoint...")
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
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            else:
                raise

        self.model.to(self.device)
        self.model.eval()

        # Initialize BPE Tokenizer
        self.model.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)
        self.tokenizer_instance = self.model.tokenizer

        # Ensure Discrete VAE is loaded (needed for teacher-forced DVAE optimization loss)
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
                if os.path.isfile(dvae_path):
                    dvae_ckpt = torch.load(dvae_path, map_location="cpu", weights_only=False)
                    state = dvae_ckpt.get("model", dvae_ckpt.get("state_dict", dvae_ckpt))
                    dvae.load_state_dict(state, strict=False)
                    dvae = dvae.to(self.device)
                    dvae.eval()
                    self.model.dvae = dvae
                    logger.info(f"XTTS Discrete VAE loaded successfully from {dvae_path}")
                else:
                    logger.warning(f"XTTS Discrete VAE file not found at {dvae_path}. Latent optimization will load DVAE dynamically if available.")
            except Exception as e:
                logger.warning(f"Could not load Discrete VAE: {e}")

        # Check if base_model exists for raw baseline comparison (un-fine-tuned)
        base_model_path = os.path.join(model_dir, "base_model.pth")
        base_config_path = os.path.join(model_dir, "base_config.json")
        if not os.path.isfile(base_config_path):
            base_config_path = config_path

        self.base_model = None
        if os.path.isfile(base_model_path):
            try:
                base_cfg = XttsConfig()
                base_cfg.load_json(base_config_path)
                base_m = Xtts.init_from_config(base_cfg)
                base_m.load_checkpoint(
                    base_cfg,
                    checkpoint_dir=model_dir,
                    checkpoint_path=base_model_path,
                    vocab_path=vocab_path,
                    eval=True,
                    use_deepspeed=False,
                    strict=False,
                )
                base_m.to(self.device)
                base_m.eval()
                base_m.tokenizer = self.model.tokenizer
                self.base_model = base_m
                logger.info(f"✓ Base XTTS Model loaded for raw baseline comparison from {base_model_path}")
            except Exception as e:
                logger.warning(f"Could not load Base XTTS Model from {base_model_path}: {e}")

        _ = self.embedding_dim
        logger.info(f"✓ Fine-Tuned XTTS Backbone loaded. Device: {self.device}, Embedding Dim: {self.embedding_dim}")

    def _ensure_loaded(self) -> None:
        if self.model is None:
            self.load_model()

    def tokenize(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Tokenize text string into token dictionary."""
        tokens = self.get_token_ids(text, language=language)
        return {"token_ids": tokens, "text": text}

    def detokenize(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back into text representation."""
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.squeeze().tolist()
        else:
            ids = list(token_ids)
        if self.tokenizer_instance and hasattr(self.tokenizer_instance, "decode"):
            return self.tokenizer_instance.decode(ids)
        return str(ids)

    def get_token_ids(self, text: str, language: str = "en") -> torch.Tensor:
        """Tokenize text into token ID tensor [1, S]."""
        self._ensure_loaded()
        if self.tokenizer_instance is not None:
            try:
                if hasattr(self.tokenizer_instance, "preprocess_text"):
                    clean_text = self.tokenizer_instance.preprocess_text(text, language)
                else:
                    clean_text = text
                token_ids = self.tokenizer_instance.encode(clean_text, lang=language)
                if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
                    token_ids = token_ids[0]
                return torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            except Exception as e:
                logger.warning(f"Tokenizer encoding failed for '{text}': {e}. Using fallback character encoding.")

        return torch.zeros((1, max(1, len(text))), dtype=torch.long, device=self.device)

    def encode_text(self, text: str, language: str = "en") -> torch.Tensor:
        """
        Encode text into continuous text embeddings c in R^[1, S, d] (d=1024).
        Used by Hopfield Memory for key computation and as base for perturbation delta.
        """
        self._ensure_loaded()
        tokens = self.get_token_ids(text, language=language)

        with torch.no_grad():
            gpt = getattr(self.model, "gpt", None)
            if gpt is not None:
                text_embed = getattr(gpt, "text_embedding", getattr(gpt, "text_embed", None))
                if text_embed is not None:
                    try:
                        emb = text_embed(tokens)
                        if isinstance(emb, tuple):
                            emb = emb[0]
                        return emb
                    except Exception as e:
                        logger.warning(f"GPT text_embedding forward failed: {e}")

                pos_embed = getattr(gpt, "text_pos_embedding", None)
                if pos_embed is not None:
                    try:
                        emb = pos_embed(tokens)
                        if isinstance(emb, tuple):
                            emb = emb[0]
                        return emb
                    except Exception:
                        pass

            if not hasattr(self, "_fallback_embed"):
                self._fallback_embed = nn.Embedding(10000, self.embedding_dim).to(self.device)
            return self._fallback_embed(tokens % 10000)

    def get_speaker_embedding(
        self,
        audio_path: Optional[str] = None,
        language: str = "en",
        ref_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract speaker conditioning (gpt_cond_latent, speaker_embedding) from reference audio.
        """
        self._ensure_loaded()
        cache_key = f"{audio_path}_{language}_{ref_text}"
        if cache_key in self._speaker_cache:
            return self._speaker_cache[cache_key]

        if not audio_path or not os.path.exists(audio_path):
            try:
                from flowedit.api.main import get_hardcoded_voice_path
                hardcoded_spk = get_hardcoded_voice_path("blessing")
                if hardcoded_spk and os.path.exists(hardcoded_spk):
                    audio_path = hardcoded_spk
                    logger.info(f"[XTTS] Defaulting to hardcoded Blessing voice: {audio_path}")
            except Exception:
                pass

        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Speaker reference audio not found: {audio_path}")

        processed_fd, processed_path = tempfile.mkstemp(suffix=".wav")
        os.close(processed_fd)

        try:
            # XTTS input conditioning expects 22050 Hz mono WAV
            if librosa is not None:
                y, sr = librosa.load(audio_path, sr=22050, mono=True)
                # High-fidelity custom voice preprocessing: trim silence and peak normalize
                if len(y) > 0:
                    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
                    if len(y_trimmed) > sr * 0.5:
                        y = y_trimmed
                    max_amp = float(np.max(np.abs(y)))
                    if max_amp > 1e-4:
                        y = (y / max_amp) * 0.95
            else:
                import torchaudio
                wav_t, orig_sr = torchaudio.load(audio_path)
                if wav_t.shape[0] > 1:
                    wav_t = torch.mean(wav_t, dim=0, keepdim=True)
                if orig_sr != 22050:
                    wav_t = torchaudio.functional.resample(wav_t, orig_sr, 22050)
                y = wav_t.squeeze(0).cpu().numpy()
                sr = 22050
                max_amp = float(np.max(np.abs(y)))
                if max_amp > 1e-4:
                    y = (y / max_amp) * 0.95

            try:
                import soundfile as sf
                sf.write(processed_path, y, 22050, subtype="PCM_16")
            except Exception:
                import wave
                int16_y = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
                with wave.open(processed_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(22050)
                    wf.writeframes(int16_y.tobytes())

            # High-fidelity conditioning parameters (up to 30s reference audio)
            gpt_cond_len = self._get_config_value("gpt_cond_len", 30)
            gpt_cond_chunk_len = self._get_config_value("gpt_cond_chunk_len", 4)
            max_ref_len = self._get_config_value("max_ref_len", 30)

            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                audio_path=[processed_path],
                gpt_cond_len=gpt_cond_len,
                gpt_cond_chunk_len=gpt_cond_chunk_len,
                max_ref_length=max_ref_len,
            )

            if not hasattr(self.model, "mel_transform") or self.model.mel_transform is None:
                import torchaudio.transforms as T
                self.model.mel_transform = T.MelSpectrogram(
                    sample_rate=22050,
                    n_fft=1024,
                    win_length=1024,
                    hop_length=256,
                    f_min=0,
                    f_max=8000,
                    n_mels=80,
                ).to(self.device)

            y_t = torch.from_numpy(y).unsqueeze(0).to(self.device).float()
            cond_mel = self.model.mel_transform(y_t)
            if cond_mel.dim() == 2:
                cond_mel = cond_mel.unsqueeze(0)
            cond_mel = torch.log(torch.clamp(cond_mel, min=1e-5))

            result = {
                "audio_path": audio_path,
                "processed_audio_path": processed_path,
                "text": ref_text or "",
                "gpt_cond_latent": gpt_cond_latent,
                "speaker_embedding": speaker_embedding,
                "cond_mel": cond_mel,
            }
            self._speaker_cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"Error computing speaker conditioning for {audio_path}: {e}", exc_info=True)
            raise
        finally:
            if os.path.exists(processed_path) and processed_path != audio_path:
                try:
                    os.remove(processed_path)
                except Exception:
                    pass

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
        """
        Compute differentiable teacher-forced cross-entropy loss for XTTS latent optimization (Paper Eq. 3).
        
        Creates a direct gradient path:
            delta -> (c + delta) -> GPT forward hook -> Cross-Entropy loss over teacher audio tokens.
        """
        self._ensure_loaded()
        gpt = getattr(self.model, "gpt", None)
        if gpt is None:
            raise RuntimeError("XTTS model does not have a GPT module loaded.")

        import torchaudio

        device = self.device
        tokens = self.get_token_ids(text, language=language)
        text_lengths = torch.tensor([tokens.shape[1]], dtype=torch.long, device=device)

        # 1. Load reference audio and compute mel-spectrogram
        waveform, sr = torchaudio.load(ref_audio_path)
        waveform = waveform.to(device)
        if sr != 22050:
            import torchaudio.transforms as T
            resampler = T.Resample(sr, 22050).to(device)
            waveform = resampler(waveform)

        target_word = kwargs.get("target_word", None)
        token_indices = kwargs.get("token_indices", None)
        ref_start_time = kwargs.get("ref_start_time", None)
        ref_end_time = kwargs.get("ref_end_time", None)

        # Trim waveform to word boundaries if provided (with generous margin to preserve consonant onsets)
        if ref_start_time is not None and ref_end_time is not None and ref_end_time > ref_start_time:
            total_dur = waveform.shape[-1] / 22050.0
            start_s = int(max(0.0, ref_start_time - 0.05) * 22050)
            end_s = int(min(total_dur, ref_end_time + 0.05) * 22050)
            if end_s > start_s + 1000:
                waveform = waveform[:, start_s:end_s]
        else:
            # Energy-based silence trimming so audio_codes capture the spoken phonemes
            non_silent = (waveform.abs() > 0.01).nonzero(as_tuple=True)
            if len(non_silent) > 1 and len(non_silent[1]) > 0:
                s_trim = max(0, non_silent[1].min().item() - 220)
                e_trim = min(waveform.shape[-1], non_silent[1].max().item() + 220)
                if e_trim > s_trim:
                    waveform = waveform[:, s_trim:e_trim]

        audio_dur = waveform.shape[-1] / 22050.0
        is_isolated = (target_word is not None and text.strip().lower() == target_word.strip().lower())
        is_word_level_ref = (target_word is not None and (audio_dur < 6.0 or is_isolated))

        all_tokens = self.get_token_ids(text, language=language)
        all_base_embeddings = self.encode_text(text, language=language)

        if is_word_level_ref and token_indices and len(token_indices) > 0 and not is_isolated:
            # Extract exact in-context target word tokens and embeddings from carrier sentence
            t_indices = sorted(list(token_indices))
            tokens = all_tokens[:, t_indices]
            text_lengths = torch.tensor([tokens.shape[1]], dtype=torch.long, device=device)
            base_slice = all_base_embeddings[:, t_indices, :]
            delta_slice = text_embedding_delta[:, t_indices, :]
            perturbed_embeddings = base_slice + delta_slice.to(base_slice.device)
        else:
            tokens = all_tokens
            text_lengths = torch.tensor([tokens.shape[1]], dtype=torch.long, device=device)
            base_embeddings = all_base_embeddings
            if text_embedding_delta.shape[1] != base_embeddings.shape[1]:
                delta_aligned = torch.zeros_like(base_embeddings)
                copy_len = min(text_embedding_delta.shape[1], base_embeddings.shape[1])
                delta_aligned[:, :copy_len, :] = text_embedding_delta[:, :copy_len, :]
            else:
                delta_aligned = text_embedding_delta
            perturbed_embeddings = base_embeddings + delta_aligned.to(base_embeddings.device)

        if not hasattr(self.model, "mel_transform") or self.model.mel_transform is None:
            import torchaudio.transforms as T
            self.model.mel_transform = T.MelSpectrogram(
                sample_rate=22050,
                n_fft=1024,
                win_length=1024,
                hop_length=256,
                f_min=0,
                f_max=8000,
                n_mels=80,
            ).to(device)

        mel = self.model.mel_transform(waveform)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        # 3. Extract discrete audio codes using DVAE
        if not hasattr(self.model, "dvae") or self.model.dvae is None:
            raise RuntimeError("XTTS DVAE is required to compute optimization loss over audio codes.")

        with torch.no_grad():
            audio_codes = self.model.dvae.get_codebook_indices(mel)
        audio_lengths = torch.tensor([audio_codes.shape[1]], dtype=torch.long, device=device)

        gpt_cond_latent = speaker_conditioning.get("gpt_cond_latent")
        if gpt_cond_latent is None:
            raise RuntimeError("Missing gpt_cond_latent in speaker_conditioning.")

        target_embed_layer = getattr(gpt, "text_embedding", getattr(gpt, "text_embed", None))
        if target_embed_layer is None:
            raise RuntimeError("Cannot find text_embedding layer in XTTS GPT.")

        hook_handle = None

        def embedding_hook(module, inputs, output):
            out_tensor = output[0] if isinstance(output, tuple) else output
            new_out = out_tensor.clone()
            t_embed = perturbed_embeddings.to(device=out_tensor.device, dtype=out_tensor.dtype)

            input_ids = inputs[0]
            L_corr = t_embed.shape[1]
            T_seq = out_tensor.shape[1]

            start_pos = -1
            t_offset = 0
            if input_ids is not None and input_ids.dim() >= 2:
                seq = tokens[0].to(input_ids.device)

                # 1. Exact full-sequence match
                best_start, best_len = -1, 0
                for i in range(input_ids.shape[1]):
                    match_len = 0
                    for j in range(min(len(seq), input_ids.shape[1] - i)):
                        if input_ids[0, i + j] == seq[j]:
                            match_len += 1
                        else:
                            break
                    if match_len > best_len:
                        best_len = match_len
                        best_start = i

                if best_start != -1 and best_len >= min(2, len(seq)):
                    start_pos = best_start
                    t_offset = 0
                elif len(seq) > 2:
                    # 2. Sub-sequence match if leading BOS differed
                    sub_seq = seq[1:]
                    for i in range(input_ids.shape[1]):
                        match_len = 0
                        for j in range(min(len(sub_seq), input_ids.shape[1] - i)):
                            if input_ids[0, i + j] == sub_seq[j]:
                                match_len += 1
                            else:
                                break
                        if match_len > best_len:
                            best_len = match_len
                            best_start = i
                    if best_start != -1 and best_len >= min(2, len(sub_seq)):
                        start_pos = best_start
                        t_offset = 1

            if start_pos == -1:
                start_pos = 0
                t_offset = 0

            avail = min(L_corr - t_offset, T_seq - start_pos)
            if avail > 0:
                new_out[:, start_pos : start_pos + avail, :] = t_embed[:, t_offset : t_offset + avail, :]

            if isinstance(output, tuple):
                return (new_out,) + output[1:]
            return new_out

        hook_handle = target_embed_layer.register_forward_hook(embedding_hook)

        cond_mel = speaker_conditioning.get("cond_mel", mel)
        if cond_mel is None:
            cond_mel = mel

        try:
            res = gpt.forward(
                text_inputs=tokens,
                text_lengths=text_lengths,
                audio_codes=audio_codes,
                wav_lengths=audio_lengths,
                cond_mels=cond_mel,
                cond_latents=gpt_cond_latent,
            )

            if isinstance(res, tuple) and len(res) > 1:
                # res[1] is the acoustic mel audio-code cross-entropy loss
                # res[0] is the text LM loss which penalizes phonetic changes; we isolate res[1]
                loss = res[1]
            elif isinstance(res, tuple):
                loss = res[0]
            elif isinstance(res, dict):
                loss = res.get("mel_loss", res.get("loss", list(res.values())[-1]))
            else:
                loss = res

            return {
                "loss": loss,
                "embedding_norm": torch.norm(perturbed_embeddings),
            }
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
        """
        Synthesize audio by injecting refined text embeddings into the GPT text embedding layer.
        """
        self._ensure_loaded()
        gpt_cond_latent = speaker_conditioning.get("gpt_cond_latent")
        speaker_emb = speaker_conditioning.get("speaker_embedding")

        if gpt_cond_latent is None or speaker_emb is None:
            logger.warning("Missing XTTS conditioning latents, falling back to direct synthesis")
            return self.synthesize_direct(text, speaker_conditioning, language=language)

        if text_embeddings is not None and text_embeddings.abs().sum() > 1e-6:
            base_embeddings = self.encode_text(text, language=language)
            delta = text_embeddings.to(base_embeddings.device) - base_embeddings
        else:
            base_embeddings = self.encode_text(text, language=language)
            delta = torch.zeros_like(base_embeddings)

        gpt = getattr(self.model, "gpt", None)
        hook_handle = None

        if gpt is not None:
            target_embed = getattr(gpt, "text_embedding", getattr(gpt, "text_embed", None))
            if target_embed is not None:
                my_tokens = self.get_token_ids(text, language=language)

                def hook(module, inputs, output):
                    out_tensor = output[0] if isinstance(output, tuple) else output
                    new_out = out_tensor.clone()
                    t_embed = text_embeddings.to(device=out_tensor.device, dtype=out_tensor.dtype)

                    input_ids = inputs[0]
                    L_corr = t_embed.shape[1]
                    T_seq = out_tensor.shape[1]

                    start_pos = -1
                    t_offset = 0
                    if input_ids is not None and input_ids.dim() >= 2:
                        seq = my_tokens[0].to(input_ids.device)

                        # 1. Exact full-sequence match
                        best_start, best_len = -1, 0
                        for i in range(input_ids.shape[1]):
                            match_len = 0
                            for j in range(min(len(seq), input_ids.shape[1] - i)):
                                if input_ids[0, i + j] == seq[j]:
                                    match_len += 1
                                else:
                                    break
                            if match_len > best_len:
                                best_len = match_len
                                best_start = i

                        if best_start != -1 and best_len >= min(2, len(seq)):
                            start_pos = best_start
                            t_offset = 0
                        elif len(seq) > 2:
                            # 2. Sub-sequence match if leading BOS differed
                            sub_seq = seq[1:]
                            for i in range(input_ids.shape[1]):
                                match_len = 0
                                for j in range(min(len(sub_seq), input_ids.shape[1] - i)):
                                    if input_ids[0, i + j] == sub_seq[j]:
                                        match_len += 1
                                    else:
                                        break
                                if match_len > best_len:
                                    best_len = match_len
                                    best_start = i
                            if best_start != -1 and best_len >= min(2, len(sub_seq)):
                                start_pos = best_start
                                t_offset = 1

                    if start_pos == -1:
                        start_pos = 0
                        t_offset = 0

                    max_avail = min(L_corr - t_offset, T_seq - start_pos)
                    if max_avail > 0:
                        new_out[:, start_pos : start_pos + max_avail, :] = t_embed[:, t_offset : t_offset + max_avail, :]
                        logger.info(
                            f"[XTTS Hook] Successfully injected refined embeddings at GPT token positions {start_pos}..{start_pos + max_avail} (delta norm: {delta.norm().item():.4f})"
                        )

                    if isinstance(output, tuple):
                        return (new_out,) + output[1:]
                    return new_out

                hook_handle = target_embed.register_forward_hook(hook)

        try:
            out = self.model.inference(
                text=text,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_emb,
                temperature=kwargs.get("temperature", 0.75),
                length_penalty=kwargs.get("length_penalty", 1.0),
                repetition_penalty=kwargs.get("repetition_penalty", 2.0),
                top_k=kwargs.get("top_k", 50),
                top_p=kwargs.get("top_p", 0.85),
                enable_text_splitting=False,
            )

            waveform = out.get("wav", None)
            if waveform is None:
                raise RuntimeError("XTTS inference returned no waveform.")

            if isinstance(waveform, torch.Tensor):
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
            else:
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
        speaker_conditioning: Dict[str, Any],
        language: str = "en",
        user_ref_text: Optional[str] = None,
        text_embedding_delta: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """
        Direct synthesis without modifications, or with direct text_embedding_delta injected.
        """
        if text_embedding_delta is not None and text_embedding_delta.abs().sum() > 1e-6:
            base_embeddings = self.encode_text(text, language=language)
            delta = text_embedding_delta.to(base_embeddings.device)
            if delta.dim() == 2:
                delta = delta.unsqueeze(0)
            B, S_target, D = base_embeddings.shape
            B_d, S_delta, D_d = delta.shape
            if S_delta != S_target:
                if S_delta > S_target:
                    delta = delta[:, :S_target, :]
                else:
                    padded_delta = torch.zeros(B, S_target, D, device=base_embeddings.device, dtype=base_embeddings.dtype)
                    padded_delta[:, :S_delta, :] = delta
                    delta = padded_delta
            refined_embeddings = base_embeddings + delta
            return self.synthesize_from_embeddings(
                text_embeddings=refined_embeddings,
                speaker_conditioning=speaker_conditioning,
                text=text,
                language=language,
                **kwargs,
            )

        self._ensure_loaded()
        gpt_cond_latent = speaker_conditioning.get("gpt_cond_latent")
        speaker_emb = speaker_conditioning.get("speaker_embedding")

        if gpt_cond_latent is None or speaker_emb is None:
            raise RuntimeError(
                "Missing XTTS conditioning latents in speaker_conditioning."
            )

        out = self.model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_emb,
            temperature=kwargs.get("temperature", 0.75),
            length_penalty=kwargs.get("length_penalty", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 2.0),
            top_k=kwargs.get("top_k", 50),
            top_p=kwargs.get("top_p", 0.85),
            enable_text_splitting=False,
        )

        waveform = out.get("wav", None)
        if waveform is None:
            raise RuntimeError("XTTS inference returned no waveform.")

        if isinstance(waveform, torch.Tensor):
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
        else:
            waveform = torch.from_numpy(np.array(waveform)).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

        sample_rate = self._get_config_value("output_sample_rate", 24000)
        return waveform, sample_rate

    def synthesize_baseline(
        self,
        text: str,
        speaker_conditioning: Dict[str, Any],
        language: str = "en",
        user_ref_text: Optional[str] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """
        Synthesize using the raw base XTTS model (un-fine-tuned, without FlowEdit corrections).
        """
        self._ensure_loaded()
        target_model = self.base_model if self.base_model is not None else self.model
        gpt_cond_latent = speaker_conditioning.get("gpt_cond_latent")
        speaker_emb = speaker_conditioning.get("speaker_embedding")

        if gpt_cond_latent is None or speaker_emb is None:
            raise RuntimeError("Missing XTTS conditioning latents in speaker_conditioning.")

        logger.info(f"[XTTS Baseline] Synthesizing raw baseline audio using {'base_model.pth' if self.base_model is not None else 'fine-tuned model'}")

        out = target_model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_emb,
            temperature=kwargs.get("temperature", 0.75),
            length_penalty=kwargs.get("length_penalty", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 2.0),
            top_k=kwargs.get("top_k", 50),
            top_p=kwargs.get("top_p", 0.85),
            enable_text_splitting=False,
        )

        waveform = out.get("wav", None)
        if waveform is None:
            raise RuntimeError("XTTS baseline inference returned no waveform.")

        if isinstance(waveform, torch.Tensor):
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
        else:
            waveform = torch.from_numpy(np.array(waveform)).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

        sample_rate = self._get_config_value("output_sample_rate", 24000)
        return waveform, sample_rate

    def _get_config_value(self, key: str, default: Any) -> Any:
        """Read a value from the XTTS configuration JSON with a fallback."""
        if self._xtts_config is not None:
            val = getattr(self._xtts_config, key, None)
            if val is not None:
                return val
            model_args = getattr(self._xtts_config, "model_args", None)
            if model_args is not None:
                if isinstance(model_args, dict):
                    val = model_args.get(key, None)
                else:
                    val = getattr(model_args, key, None)
                if val is not None:
                    return val
        return default
