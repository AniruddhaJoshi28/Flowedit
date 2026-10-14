"""
Contextual Sense Encoder for EMA homograph disambiguation.

Produces one L2-normalised vector per word occurrence from the middle hidden layers of a
pretrained text encoder. Each occurrence is encoded inside its own clause-bounded word window,
so two occurrences of the same spelling in one sentence ("contains lead, ... the lead
researcher") receive independent context vectors instead of a shared sentence context.

The vectors live in a backbone-independent space: sense prototypes learned while correcting
XTTS are directly usable by F5-TTS (and vice versa), unlike backbone text embeddings whose
spaces and dimensions differ.
"""

import logging
import re
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ContextualSenseEncoder:
    """Lazily-loaded occurrence encoder with an LRU cache keyed by (text, spans)."""

    def __init__(
        self,
        model_name: str,
        layers: Sequence[int],
        context_words: int,
        clause_delimiters: str,
        device: str = "cpu",
        cache_size: int = 256,
    ):
        self.model_name = (model_name or "").strip()
        self.layers = tuple(int(i) for i in layers)
        self.context_words = max(1, int(context_words))
        self.clause_pattern = re.compile("[" + re.escape(clause_delimiters) + "]") if clause_delimiters else None
        self.device = device
        self.cache_size = max(0, int(cache_size))
        self._tokenizer = None
        self._model = None
        self._load_failed = not self.model_name
        self._lock = threading.RLock()  # guards model loading, forward passes and the LRU cache
        self._cache: "OrderedDict[Tuple[str, Tuple[Tuple[int, int], ...]], torch.Tensor]" = OrderedDict()

    @property
    def available(self) -> bool:
        return self._ensure_loaded()

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        with self._lock:
            if self._model is not None:
                return True
            try:
                tok, mdl = _load_model(self.model_name, self.device)
                self._tokenizer = tok
                self._model = mdl
                n_hidden = int(getattr(mdl.config, "num_hidden_layers", max(self.layers)))
                self.layers = tuple(i for i in self.layers if 0 <= i <= n_hidden) or (n_hidden,)
                logger.info(f"✓ Contextual sense encoder ready: {self.model_name} (layers={self.layers}, device={self.device})")
                return True
            except Exception as e:
                self._load_failed = True
                logger.warning(
                    f"Contextual sense encoder '{self.model_name}' unavailable ({e}). "
                    f"Homograph disambiguation falls back to backbone context keys."
                )
                return False

    def _window(self, text: str, char_start: int, char_end: int) -> Tuple[str, int, int]:
        """±N word window around [char_start, char_end); returns (snippet, start, end).

        The window stops at clause punctuation and at neighbouring occurrences of the same word,
        whose surrounding words describe their own sense rather than this one's.
        """
        lo, hi = 0, len(text)
        if self.clause_pattern is not None:
            for m in self.clause_pattern.finditer(text):
                if m.end() <= char_start:
                    lo = m.end()
                elif m.start() >= char_end:
                    hi = m.start()
                    break
        target = text[char_start:char_end].strip()
        if target:
            for m in re.finditer(r"\b" + re.escape(target) + r"\b", text, flags=re.IGNORECASE):
                if m.end() <= char_start:
                    lo = max(lo, m.end())
                elif m.start() >= char_end:
                    hi = min(hi, m.start())
                    break
        words = [m for m in re.finditer(r"\S+", text) if m.end() > lo and m.start() < hi]
        hit = [i for i, m in enumerate(words) if m.start() < char_end and m.end() > char_start]
        if not hit:
            return text, char_start, char_end
        sel = words[max(0, hit[0] - self.context_words): hit[-1] + self.context_words + 1]
        s = max(lo, sel[0].start())
        e = min(hi, sel[-1].end())
        s = min(s, char_start)
        e = max(e, char_end)
        return text[s:e], char_start - s, char_end - s

    def encode_batch(self, occurrences: Sequence[Tuple[str, int, int]]) -> torch.Tensor:
        """Encode (text, start, end) occurrences from any texts in one forward pass → [N, H] (uncached)."""
        if not self._ensure_loaded():
            raise RuntimeError("contextual sense encoder unavailable")
        windows = [self._window(t, int(a), int(b)) for t, a, b in occurrences]
        return self._forward(windows)

    def _forward(self, windows) -> torch.Tensor:
        with self._lock, torch.no_grad():
            enc = self._tokenizer(
                [w[0] for w in windows],
                return_offsets_mapping=True,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self._model(**enc, output_hidden_states=True)
            hidden = torch.stack([out.hidden_states[i] for i in self.layers]).mean(dim=0).float().cpu()

        vectors: List[torch.Tensor] = []
        for row, (_, cs, ce) in enumerate(windows):
            off = offsets[row]
            sel = (off[:, 1] > off[:, 0]) & (off[:, 0] < ce) & (off[:, 1] > cs)
            if not bool(sel.any()):
                sel = off[:, 1] > off[:, 0]
            vectors.append(F.normalize(torch.nan_to_num(hidden[row][sel].mean(dim=0)), p=2, dim=-1))
        return torch.stack(vectors)

    def encode(self, text: str, spans: Sequence[Tuple[int, int]]) -> Optional[torch.Tensor]:
        """Encode each character span of `text` → [len(spans), H] normalised, or None if unavailable."""
        if not spans or not text or not self._ensure_loaded():
            return None
        key = (text, tuple((int(a), int(b)) for a, b in spans))
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached

        windows = [self._window(text, int(a), int(b)) for a, b in spans]
        result = self._forward(windows)

        if self.cache_size:
            with self._lock:
                self._cache[key] = result
                if len(key[1]) > 1:
                    # Per-occurrence entries let later single-span lookups reuse this batch
                    for i, span in enumerate(key[1]):
                        self._cache[(text, (span,))] = result[i:i + 1]
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        return result


_ENCODERS: Dict[Tuple, ContextualSenseEncoder] = {}
_ENCODERS_LOCK = threading.Lock()
_MODELS: Dict[Tuple[str, str], Tuple] = {}


def _load_model(model_name: str, device: str):
    """(tokenizer, model) shared by every encoder using the same weights on the same device."""
    with _ENCODERS_LOCK:
        cached = _MODELS.get((model_name, device))
        if cached is not None:
            return cached
        from transformers import AutoModel, AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            mdl = AutoModel.from_pretrained(model_name, local_files_only=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(model_name)
            mdl = AutoModel.from_pretrained(model_name)
        if not getattr(tok, "is_fast", False):
            raise RuntimeError("a fast tokenizer (offset mapping) is required")
        _MODELS[(model_name, device)] = (tok, mdl.to(device).eval())
        return _MODELS[(model_name, device)]


def get_sense_encoder(config, context_words: Optional[int] = None) -> Optional[ContextualSenseEncoder]:
    """Process-wide shared encoder for a MemoryConfig (one model instance for XTTS and F5-TTS).

    `context_words` overrides the window size; encoders with different windows share weights.
    """
    model_name = (getattr(config, "sense_encoder_model", "") or "").strip()
    if not model_name:
        return None
    sig = (
        model_name,
        tuple(getattr(config, "sense_encoder_layers", ())),
        int(context_words if context_words is not None else getattr(config, "sense_context_words", 6)),
        getattr(config, "sense_clause_delimiters", ""),
        getattr(config, "sense_encoder_device", "cpu"),
        int(getattr(config, "sense_cache_size", 256)),
    )
    with _ENCODERS_LOCK:
        enc = _ENCODERS.get(sig)
        if enc is None:
            enc = ContextualSenseEncoder(*sig)
            _ENCODERS[sig] = enc
        return enc
