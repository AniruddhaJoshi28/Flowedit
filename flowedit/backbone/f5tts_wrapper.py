"""
F5-TTS Backbone Implementation for FlowEdit.

Paper Reference: FlowEdit (arXiv:2606.20518), Section 3.1 & 3.2.
Continuous Flow-Matching (CFM) Diffusion Transformer (DiT) text-to-speech backbone.
"""

import os
import re
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

    @property
    def is_character_level(self) -> bool:
        return True

    @property
    def delta_bounds(self) -> Tuple[float, float]:
        return float(self.config.f5tts_delta_clamp), float(self.config.f5tts_delta_norm_cap)

    def _resolve_f5_paths(self) -> Tuple[str, str, str, str]:
        """Automatically discover F5-TTS checkpoint, vocab, vocoder files, and model architecture name.
        
        Strictly prioritizes locally downloaded models in `models/f5tts` and `models/` over any Hugging Face hub cache,
        preventing unexpected downloads and resolving server disk space issues.
        """
        ckpt_file = getattr(self.config, 'f5tts_ckpt_file', "") or os.environ.get("FLOWEDIT_F5TTS_CKPT", "")
        vocab_file = getattr(self.config, 'f5tts_vocab_file', "") or os.environ.get("FLOWEDIT_F5TTS_VOCAB", "")
        vocoder_local_path = getattr(self.config, 'vocoder_local_path', "") or os.environ.get("FLOWEDIT_VOCODER_DIR", "")
        model_name = getattr(self.config, 'f5tts_model_name', "") or os.environ.get("FLOWEDIT_F5TTS_MODEL", "")

        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        flowedit_root = os.path.abspath(os.path.join(pkg_dir, "..", ".."))
        workspace_root = os.path.abspath(os.path.join(flowedit_root, ".."))

        search_dirs = [
            os.path.join(flowedit_root, "models", "f5tts"),
            os.path.join(flowedit_root, "models", "F5TTS"),
            os.path.join(flowedit_root, "models", "F5TTS_Base"),
            os.path.join(flowedit_root, "models", "F5TTS_v1_Base"),
            os.path.join(workspace_root, "models", "f5tts"),
            os.path.join(workspace_root, "models", "F5TTS"),
            "/home/rsurya/projects/flow_edit/Flowedit/models/f5tts",
            "/home/rsurya/projects/flow_edit/Flowedit/models/F5TTS_Base",
            "/home/rsurya/projects/flow_edit/Flowedit/models/F5TTS_v1_Base",
            "/home/rsurya/projects/flow_edit/models/f5tts",
            os.path.expanduser("~/projects/flow_edit/Flowedit/models/f5tts"),
            os.path.expanduser("~/projects/flow_edit/models/f5tts"),
            os.path.join(flowedit_root, "models"),
            os.path.join(workspace_root, "models"),
        ]

        # 1. Discover local F5-TTS checkpoint (recursive search in local models)
        if not ckpt_file or not os.path.isfile(ckpt_file):
            discovered_ckpts = []
            seen_ckpt_paths = set()
            for s_dir in search_dirs:
                if s_dir and os.path.isdir(s_dir):
                    for root, dirs, files in os.walk(s_dir):
                        # Avoid descending into XTTS or Vocos folders
                        if any(x in root.lower() for x in ("xtts", "vocos")):
                            continue
                        for f in files:
                            f_lower = f.lower()
                            if f_lower.endswith((".safetensors", ".pt", ".bin", ".ckpt")):
                                f_path = os.path.abspath(os.path.join(root, f))
                                if f_path in seen_ckpt_paths:
                                    continue
                                seen_ckpt_paths.add(f_path)
                                try:
                                    f_size = os.path.getsize(f_path)
                                    if f_size > 50_000_000:  # > 50 MB model weight
                                        discovered_ckpts.append((f_path, f_size, f_lower))
                                except Exception:
                                    pass

            if discovered_ckpts:
                # Rank candidates: prioritize 1250000 or 1200000, safetensors over pt
                def _rank_ckpt(item):
                    path, size, name = item
                    score = 0
                    if "1250000" in name:
                        score += 40
                    elif "1200000" in name:
                        score += 35
                    if "safetensors" in name:
                        score += 10
                    if "f5tts" in path.lower() or "f5" in path.lower():
                        score += 5
                    return score

                discovered_ckpts.sort(key=_rank_ckpt, reverse=True)
                ckpt_file = os.path.abspath(discovered_ckpts[0][0])
                logger.info(f"✓ Discovered local F5-TTS checkpoint in project: {ckpt_file} ({discovered_ckpts[0][1] / (1024 * 1024):.1f} MB)")

        # 2. Discover local vocabulary
        if not vocab_file or not os.path.isfile(vocab_file):
            discovered_vocabs = []
            # Check directory containing ckpt_file first
            if ckpt_file:
                ckpt_dir = os.path.dirname(ckpt_file)
                cand_v = os.path.join(ckpt_dir, "vocab.txt")
                if os.path.isfile(cand_v) and os.path.getsize(cand_v) > 50:
                    discovered_vocabs.append(cand_v)

            for s_dir in search_dirs:
                if s_dir and os.path.isdir(s_dir):
                    for root, dirs, files in os.walk(s_dir):
                        for f in files:
                            if f.lower() == "vocab.txt":
                                v_path = os.path.join(root, f)
                                if os.path.isfile(v_path) and os.path.getsize(v_path) > 50:
                                    discovered_vocabs.append(v_path)

            cand_emilia = os.path.join(flowedit_root, "data", "Emilia_ZH_EN_pinyin", "vocab.txt")
            if os.path.isfile(cand_emilia):
                discovered_vocabs.append(cand_emilia)

            if discovered_vocabs:
                vocab_file = os.path.abspath(discovered_vocabs[0])
                logger.info(f"✓ Discovered local F5-TTS vocab: {vocab_file}")

        # 3. Discover local Vocos vocoder
        if not vocoder_local_path or not os.path.isdir(vocoder_local_path):
            vocoder_candidates = [
                os.path.join(flowedit_root, "models", "vocos"),
                os.path.join(flowedit_root, "models", "f5tts", "vocos"),
                os.path.join(workspace_root, "models", "vocos"),
                "/home/rsurya/projects/flow_edit/Flowedit/models/vocos",
                "/home/rsurya/projects/flow_edit/models/vocos",
                os.path.expanduser("~/projects/flow_edit/Flowedit/models/vocos"),
                os.path.expanduser("~/projects/flow_edit/models/vocos"),
            ]
            for cand in vocoder_candidates:
                if cand and os.path.isdir(cand):
                    if os.path.isfile(os.path.join(cand, "config.yaml")):
                        vocoder_local_path = os.path.abspath(cand)
                        break
                    # Check immediate subdirectories
                    try:
                        for sub in os.listdir(cand):
                            sub_p = os.path.join(cand, sub)
                            if os.path.isdir(sub_p) and os.path.isfile(os.path.join(sub_p, "config.yaml")):
                                vocoder_local_path = os.path.abspath(sub_p)
                                break
                    except Exception:
                        pass
                if vocoder_local_path:
                    logger.info(f"✓ Discovered local Vocos vocoder: {vocoder_local_path}")
                    break

        # 4. Fallback to HF hub cache ONLY if no local checkpoint was found anywhere
        if not ckpt_file or not os.path.isfile(ckpt_file):
            logger.warning("No local checkpoint found in models/f5tts. Checking Hugging Face cache fallback...")
            try:
                hf_hub = os.path.expanduser("~/.cache/huggingface/hub/models--SWivid--F5-TTS/snapshots")
                if os.path.isdir(hf_hub):
                    for snap in os.listdir(hf_hub):
                        snap_dir = os.path.join(hf_hub, snap)
                        cand_v1 = os.path.join(snap_dir, "F5TTS_v1_Base", "model_1250000.safetensors")
                        if os.path.isfile(cand_v1):
                            ckpt_file = os.path.abspath(cand_v1)
                            break
                        cand_base = os.path.join(snap_dir, "F5TTS_Base", "model_1200000.safetensors")
                        if os.path.isfile(cand_base):
                            ckpt_file = os.path.abspath(cand_base)
                            break
            except Exception:
                pass

        # 5. Automatically align model architecture name with checkpoint
        if not model_name:
            if ckpt_file:
                ckpt_lower = ckpt_file.lower()
                if "1250000" in ckpt_lower or "f5tts_v1" in ckpt_lower or "v1" in ckpt_lower:
                    model_name = "F5TTS_v1_Base"
                elif "1200000" in ckpt_lower or "f5tts_base" in ckpt_lower or "base" in ckpt_lower:
                    model_name = "F5TTS_Base"
                else:
                    model_name = "F5TTS_Base"
            else:
                model_name = "F5TTS_Base"

        # 6. When local models are found, enforce offline mode to prevent Hugging Face from downloading or using space
        if ckpt_file and os.path.isfile(ckpt_file):
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            logger.info("✓ F5-TTS Offline Mode active: using 100% local models (HF_HUB_OFFLINE=1, space-safe)")

        return ckpt_file, vocab_file, vocoder_local_path, model_name

    def load_model(self) -> None:
        """Load F5-TTS model, tokenizer vocab, and vocoder."""
        logger.info(f"Loading F5-TTS backbone on device '{self.device}'...")
        try:
            from f5_tts.api import F5TTS
            import f5_tts.api as f5_api

            ckpt_file, vocab_file, vocoder_local_path, model_name = self._resolve_f5_paths()

            import inspect
            sig = inspect.signature(F5TTS.__init__)
            params = sig.parameters

            init_kwargs = {}
            if "model" in params:
                init_kwargs["model"] = model_name
            elif "model_type" in params:
                init_kwargs["model_type"] = model_name
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

            logger.info(f"Initializing F5TTS (architecture: {model_name}, ckpt: {ckpt_file or 'Auto-Cached'}, vocab: {vocab_file or 'Default'})")

            try:
                self.tts_api = F5TTS(**init_kwargs)
            except Exception as e_first:
                logger.warning(f"F5TTS initialization with custom paths failed ({e_first}). Retrying with fallback...")
                # Remove custom local paths and retry, preserving model architecture name
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

    @staticmethod
    def _preset_prompt_text(audio_ref: Optional[str], prompt_text: str) -> str:
        """Swap a preset voice's full transcript for the trimmed clause its F5 prompt audio contains."""
        from flowedit.utils.voice_finder import get_preset_voice_transcript
        clause = get_preset_voice_transcript(str(audio_ref or ""), for_f5=True)
        if not clause or not prompt_text:
            return prompt_text
        norm = lambda t: re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()
        if norm(prompt_text) != norm(clause) and norm(prompt_text).startswith(norm(clause)):
            return clause
        return prompt_text

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

    @staticmethod
    def _find_clean_clause_boundary(
        waveform: torch.Tensor,
        sr: int = 24000,
        search_start_sec: float = 4.5,
        search_end_sec: float = 7.5,
        silence_thresh: float = 0.008,
    ) -> int:
        """Dynamically detect inter-clause silence pause to prevent bleeding into subsequent words."""
        wav_1d = waveform.squeeze(0).abs()
        total_samples = wav_1d.shape[-1]
        s_start = int(search_start_sec * sr)
        s_end = min(total_samples, int(search_end_sec * sr))
        if s_end <= s_start:
            return total_samples

        win = int(0.04 * sr)   # 40ms window
        hop = int(0.01 * sr)   # 10ms hop
        min_rms = 1e9
        best_sample = s_end

        # Find sustained silence valley below threshold (energy drops after clause 1 speech)
        for i in range(s_start, s_end - win * 2, hop):
            r1 = torch.sqrt(torch.mean(wav_1d[i : i + win] ** 2)).item()
            r2 = torch.sqrt(torch.mean(wav_1d[i + win : i + 2 * win] ** 2)).item()
            combined_rms = max(r1, r2)
            if combined_rms < silence_thresh:  # Sustained inter-clause silence
                best_sample = i + int(0.03 * sr)
                break
            if combined_rms < min_rms:
                min_rms = combined_rms
                best_sample = i + int(0.03 * sr)

        return min(total_samples, max(s_start, best_sample))

    def get_speaker_embedding(
        self,
        audio_path: Optional[str] = None,
        language: str = "en",
        ref_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract speaker conditioning from reference audio, ensuring clean verbatim prompt for F5-TTS."""
        prepared_audio = validate_reference_audio(audio_path, target_sample_rate=24000)

        cache_key = f"{prepared_audio.checksum}_{language}_{ref_text}"
        if cache_key in self._speaker_cache:
            return self._speaker_cache[cache_key]

        # 1. Resolve ground-truth prompt text (critical: F5-TTS DiT requires exact matching prompt text)
        name_lower = os.path.basename(str(audio_path or "")).lower()
        is_blessing = "blessing" in name_lower or "female" in name_lower or "default" in name_lower
        is_michael = "michael" in name_lower or "male" in name_lower or "man" in name_lower

        from flowedit.utils.voice_finder import get_preset_voice_transcript
        f5_preset_text = get_preset_voice_transcript(audio_path, for_f5=True) or ""

        prompt_text = ref_text.strip() if ref_text and ref_text.strip() else ""
        if not prompt_text:
            prompt_text = f5_preset_text or ""
        else:
            prompt_text = self._preset_prompt_text(audio_path, prompt_text)

        # 2. For Blessing and Michael presets, trim to exact clean clause to match verbatim in-context prompt
        ref_waveform = prepared_audio.waveform

        if is_blessing and ref_waveform.shape[-1] > int(4.5 * 24000):
            # Clause 1 ends at ~4.25s. Detect pause between clause 1 and clause 2
            cut_sample = self._find_clean_clause_boundary(ref_waveform, 24000, search_start_sec=4.2, search_end_sec=5.0, silence_thresh=0.018)
            ref_waveform = ref_waveform[:, :cut_sample]
            fade_len = min(int(0.030 * 24000), ref_waveform.shape[-1])
            fade = torch.linspace(1.0, 0.0, fade_len, device=ref_waveform.device)
            ref_waveform[:, -fade_len:] = ref_waveform[:, -fade_len:] * fade
            # Append 250ms of pure digital silence to guarantee complete acoustic isolation across boundary
            silence_pad = torch.zeros(ref_waveform.shape[0], int(0.250 * 24000), device=ref_waveform.device, dtype=ref_waveform.dtype)
            ref_waveform = torch.cat([ref_waveform, silence_pad], dim=-1)
            logger.info(f"Trimmed Blessing reference audio to clean {cut_sample / 24000.0:.2f}s clause with 250ms silence pad.")
        elif is_michael and ref_waveform.shape[-1] > int(3.5 * 24000):
            # Clause 1 'The sensibility of the palm remains normal.' ends cleanly at ~2.90s.
            # Detect silence valley between 2.95s and 3.35s
            cut_sample = self._find_clean_clause_boundary(ref_waveform, 24000, search_start_sec=2.95, search_end_sec=3.35, silence_thresh=0.008)
            ref_waveform = ref_waveform[:, :cut_sample]
            fade_len = min(int(0.030 * 24000), ref_waveform.shape[-1])
            fade = torch.linspace(1.0, 0.0, fade_len, device=ref_waveform.device)
            ref_waveform[:, -fade_len:] = ref_waveform[:, -fade_len:] * fade
            # Append 250ms of pure digital silence so Michael prompt never leaks or regenerates 'branches'/'rangers'
            silence_pad = torch.zeros(ref_waveform.shape[0], int(0.250 * 24000), device=ref_waveform.device, dtype=ref_waveform.dtype)
            ref_waveform = torch.cat([ref_waveform, silence_pad], dim=-1)
            logger.info(f"Trimmed Michael reference audio to clean {cut_sample / 24000.0:.2f}s clause ('The sensibility of the palm remains normal.') with 250ms silence pad.")
        elif ref_waveform.shape[-1] > int(7.0 * 24000):
            # Trim generic custom audio at the nearest natural silence valley around 4.8s - 6.8s
            cut_sample = self._find_clean_clause_boundary(ref_waveform, 24000, search_start_sec=4.8, search_end_sec=6.8, silence_thresh=0.010)
            ref_waveform = ref_waveform[:, :cut_sample]
            fade_len = min(int(0.030 * 24000), ref_waveform.shape[-1])
            fade = torch.linspace(1.0, 0.0, fade_len, device=ref_waveform.device)
            ref_waveform[:, -fade_len:] = ref_waveform[:, -fade_len:] * fade
            silence_pad = torch.zeros(ref_waveform.shape[0], int(0.250 * 24000), device=ref_waveform.device, dtype=ref_waveform.dtype)
            ref_waveform = torch.cat([ref_waveform, silence_pad], dim=-1)
            logger.info(f"Trimmed custom reference audio to clean {cut_sample / 24000.0:.2f}s clause with 250ms silence pad.")

        # Ensure clean audio normalization without digital clipping
        ref_np = ref_waveform.squeeze(0).cpu().numpy()
        max_val = float(np.max(np.abs(ref_np)))
        if max_val > 0.98:
            ref_np = ref_np * (0.95 / max_val)

        processed_fd, processed_path = tempfile.mkstemp(suffix=".wav")
        os.close(processed_fd)
        sf.write(processed_path, ref_np, 24000)

        # Flush stale caches in F5-TTS infer utilities to guarantee newly trimmed audio is used
        try:
            import f5_tts.infer.utils_infer as u_inf
            if hasattr(u_inf, "_ref_audio_cache") and isinstance(u_inf._ref_audio_cache, dict):
                u_inf._ref_audio_cache.clear()
            if hasattr(u_inf, "_ref_text_cache") and isinstance(u_inf._ref_text_cache, dict):
                u_inf._ref_text_cache.clear()
        except Exception:
            pass

        # 3. If still empty (custom audio upload), perform safe transcription fallback
        if not prompt_text:
            try:
                import faster_whisper
                fw_model = faster_whisper.WhisperModel("base", device="cpu", compute_type="int8")
                segs, _ = fw_model.transcribe(processed_path, language=language)
                prompt_text = " ".join(s.text.strip() for s in segs).strip()
            except Exception:
                try:
                    import whisperx
                    device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
                    compute_type = "float16" if device == "cuda" else "int8"
                    _whisper = whisperx.load_model("base", device=device, compute_type=compute_type)
                    audio_np, _ = sf.read(processed_path)
                    ref_result = _whisper.transcribe(audio_np, language=language)
                    if "segments" in ref_result and ref_result["segments"]:
                        prompt_text = " ".join([seg["text"] for seg in ref_result["segments"]]).strip()
                except Exception as e:
                    logger.warning(f"Reference audio transcription notice: {e}")

        if not prompt_text or prompt_text == ".":
            prompt_text = f5_preset_text or get_preset_voice_transcript("default", for_f5=True) or "."

        res = {
            "audio_path": prepared_audio.source_path,
            "processed_audio_path": processed_path,
            "text": prompt_text,
            "duration_seconds": float(ref_waveform.shape[-1]) / 24000.0,
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
                if delta_cast.shape[-1] != raw_out.shape[-1]:
                    flat_d = delta_cast.reshape(-1, 1, delta_cast.shape[-1]).float()
                    res_d = F.interpolate(flat_d, size=raw_out.shape[-1], mode='linear', align_corners=True)
                    delta_cast = res_d.reshape(list(delta_cast.shape[:-1]) + [raw_out.shape[-1]]).to(dtype=raw_out.dtype)
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

        # A preset voice's full transcript does not match its trimmed F5 prompt audio
        ref_text = self._preset_prompt_text(speaker_conditioning.get("audio_path", "") or temp_ref, ref_text)

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
            delta_rows = text_embedding_delta.detach()
            if delta_rows.dim() == 2:
                delta_rows = delta_rows.unsqueeze(0)
            spans = self._active_delta_spans(delta_rows)
            clamp_val = self.delta_bounds[0]
            placements_cache: Dict[Tuple[int, ...], List[Tuple[int, int, int]]] = {}

            def hook(module, inputs, output):
                if not inputs or not isinstance(inputs[0], torch.Tensor) or not spans:
                    return output
                ids_t = inputs[0]
                # Unconditional CFG pass: F5-TTS drops text by feeding all-zero tokens
                if (ids_t == 0).all().item():
                    return output

                with torch.set_grad_enabled(False):
                    is_tuple = isinstance(output, tuple)
                    raw_out = output[0] if is_tuple else output
                    t_delta = delta_rows.to(device=raw_out.device, dtype=raw_out.dtype)
                    if t_delta.shape[-1] != raw_out.shape[-1]:
                        flat_d = t_delta.reshape(-1, 1, t_delta.shape[-1]).float()
                        res_d = F.interpolate(flat_d, size=raw_out.shape[-1], mode='linear', align_corners=True)
                        t_delta = res_d.reshape(list(t_delta.shape[:-1]) + [raw_out.shape[-1]]).to(dtype=raw_out.dtype)
                    t_delta = torch.clamp(t_delta, -clamp_val, clamp_val)

                    out = raw_out.clone()
                    injected = []
                    for b in range(ids_t.shape[0]):
                        row = ids_t[b].tolist()
                        if not any(row):
                            continue  # batched unconditional row
                        sig = tuple(row)
                        places = placements_cache.get(sig)
                        if places is None:
                            places = self._locate_delta_spans(row, target_ids, spans)
                            placements_cache[sig] = places
                        for src_s, src_e, dst_s in places:
                            n = min(src_e - src_s, raw_out.shape[1] - dst_s)
                            if n > 0:
                                out[b, dst_s:dst_s + n, :] += t_delta[0, src_s:src_s + n, :]
                                injected.append((src_s, dst_s, n))

                    if injected:
                        logger.info(
                            f"✓ [F5TTS Span Injection] {module.__class__.__name__}: (text_pos→input_pos, len) "
                            f"{[(s_, d_, n_) for s_, d_, n_ in injected]}"
                        )
                    if is_tuple:
                        return (out,) + output[1:]
                    return out

            for target in hook_targets:
                hook_handles.append(target.register_forward_hook(hook))
            logger.info(f"[F5TTS Injection] Registered forward hook across {len(hook_targets)} target module(s).")

        # Determine natural speech speed (default 0.85 for relaxed, clear pacing)
        speed_val = kwargs.get("speed")
        if speed_val is None:
            env_speed = os.environ.get("FLOWEDIT_F5TTS_SPEED")
            if env_speed:
                try:
                    speed_val = float(env_speed)
                except ValueError:
                    speed_val = None
        if speed_val is None:
            if planned and hasattr(planned, "dynamic_speed_factor") and planned.dynamic_speed_factor:
                speed_val = min(float(planned.dynamic_speed_factor), 0.88)
            else:
                speed_val = 0.85

        # Determine deterministic seed for flow-matching ODE integration (Paper Section 3.2)
        seed_val = kwargs.get("seed")
        if seed_val is None:
            env_seed = os.environ.get("FLOWEDIT_F5TTS_SEED")
            if env_seed:
                try:
                    seed_val = int(env_seed)
                except ValueError:
                    seed_val = 42
            else:
                seed_val = 42
        else:
            try:
                seed_val = int(seed_val)
            except ValueError:
                seed_val = 42

        torch.manual_seed(seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_val)

        nfe_val = int(kwargs.get("nfe_step", 32))
        cfg_val = float(kwargs.get("cfg_strength", 2.0))

        logger.info(
            f"[F5TTS Synthesize] Deterministic flow matching active "
            f"(seed={seed_val}, nfe_step={nfe_val}, cfg_strength={cfg_val:.2f}, speed={float(speed_val):.2f})"
        )

        infer_kwargs = {
            "ref_file": temp_ref,
            "ref_text": ref_text.strip(),
            "gen_text": text.strip(),
            "speed": float(speed_val),
            "nfe_step": nfe_val,          # 32 ODE steps
            "cfg_strength": cfg_val,     # Standard F5-TTS CFG
            "target_rms": 0.1,
            "sway_sampling_coef": -1.0,
            "remove_silence": True,
            "seed": seed_val,
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

                # Amplitude safety normalization between -0.95 and +0.95 to eliminate digital clipping
                max_amp = float(wav.abs().max().item())
                if max_amp > 0.98:
                    wav = wav * (0.95 / max_amp)

                # Clean leading onset: remove excessive dead frames while preserving 35ms natural pre-speech lead-in
                wav_1d = wav.squeeze(0).abs()
                energy_thresh = 0.003
                active_indices = torch.where(wav_1d > energy_thresh)[0]
                if len(active_indices) > 0:
                    first_active = active_indices[0].item()
                    lead_silence = int(0.035 * sr)
                    trim_start = max(0, first_active - lead_silence)
                    if trim_start > int(0.020 * sr):
                        wav = wav[:, trim_start:]

                # Gentle 10ms raised-cosine fade-in on leading onset to prevent any transient click or boundary pop
                fade_in_len = min(int(0.010 * sr), wav.shape[-1])
                if fade_in_len > 0:
                    fade_in = 0.5 * (1.0 - torch.cos(torch.linspace(0.0, torch.pi, fade_in_len, device=wav.device)))
                    wav[:, :fade_in_len] = wav[:, :fade_in_len] * fade_in
                return wav, sr
            return torch.zeros(1, 24000), 24000

        finally:
            for h in hook_handles:
                h.remove()

    @staticmethod
    def _active_delta_spans(delta: torch.Tensor) -> List[Tuple[int, int]]:
        """Contiguous runs of non-zero perturbation rows in a [1, L, d] delta."""
        active = (delta[0].abs().sum(dim=-1) > 1e-8).tolist()
        spans, start = [], None
        for i, a in enumerate(active + [False]):
            if a and start is None:
                start = i
            elif not a and start is not None:
                spans.append((start, i))
                start = None
        return spans

    @staticmethod
    def _locate_delta_spans(
        input_ids: List[int],
        target_ids: List[int],
        spans: List[Tuple[int, int]],
        max_context: int = 8,
    ) -> List[Tuple[int, int, int]]:
        """Map each corrected span of the generation text onto the model's input token row.

        The row holds the reference prompt followed by (a chunk of) the generation text, with
        F5-TTS's +1 vocabulary shift applied before the embedding table. Each span is anchored by
        its own characters plus up to `max_context` neighbours on each side, so placement stays
        exact when the text is chunked or the same word occurs several times.
        """
        def positions(seq: List[int], pat: List[int]) -> List[int]:
            m = len(pat)
            return [i for i in range(len(seq) - m + 1) if seq[i:i + m] == pat]

        n_t = len(target_ids)
        placements = []
        for s, e in spans:
            found = None
            for k in sorted({max_context, max_context // 2, 2, 0}, reverse=True):
                lo, hi = max(0, s - k), min(n_t, e + k)
                pattern = target_ids[lo:hi]
                if not pattern:
                    continue
                in_target = positions(target_ids, pattern)
                for shift in (1, 0):
                    hits = positions(input_ids, [t + shift for t in pattern])
                    if not hits:
                        continue
                    if len(hits) == 1 and len(in_target) == 1:
                        found = hits[0] + (s - lo)
                    elif len(hits) == len(in_target):
                        # Whole repeated context present: the occurrence rank identifies the span
                        found = hits[in_target.index(lo)] + (s - lo)
                    break  # pattern present but ambiguous: a shorter context cannot disambiguate
                if found is not None or any(positions(input_ids, [t + sh for t in pattern]) for sh in (1, 0)):
                    break
            if found is not None:
                placements.append((s, e, found))
        return placements

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
