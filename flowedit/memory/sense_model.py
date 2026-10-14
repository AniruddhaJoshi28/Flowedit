"""
Fine-tuned homograph sense classifier (encoder trained end-to-end; see scripts/train_sense_model.py).

Each occurrence is encoded inside its own clause-bounded window (stopping at neighbouring
occurrences of the same word); the target word's token states feed one linear layer over all
senses, masked to the senses of that word.
"""

import json
import logging
import os
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from flowedit.memory.sense_encoder import ContextualSenseEncoder

logger = logging.getLogger(__name__)


class SenseModel(nn.Module):
    def __init__(self, encoder, tokenizer, labels: List[str], word_labels: Dict[str, List[str]], window: int, delimiters: str):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.labels = list(labels)
        self.word_labels = {w: list(v) for w, v in word_labels.items()}
        self.window = int(window)
        self.delimiters = delimiters
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(encoder.config.hidden_size, len(labels))
        self._windower = ContextualSenseEncoder("", (0,), self.window, delimiters, cache_size=0)
        index = {l: i for i, l in enumerate(self.labels)}
        self._masks = {w: torch.tensor([l in set(v) for l in self.labels]) for w, v in self.word_labels.items()}
        self._index = index

    @classmethod
    def from_base(cls, base: str, labels, word_labels, window: int, delimiters: str) -> "SenseModel":
        from transformers import AutoModel, AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
            enc = AutoModel.from_pretrained(base, local_files_only=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(base)
            enc = AutoModel.from_pretrained(base)
        return cls(enc, tok, labels, word_labels, window, delimiters)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SenseModel":
        from transformers import AutoModel, AutoTokenizer
        with open(os.path.join(path, "sense_labels.json"), encoding="utf-8") as f:
            meta = json.load(f)
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        enc = AutoModel.from_pretrained(path, local_files_only=True, torch_dtype=torch.float32)
        model = cls(enc, tok, meta["labels"], meta["word_labels"], meta["window"], meta["delimiters"])
        model.head.load_state_dict(torch.load(os.path.join(path, "sense_head.pt"), map_location="cpu"))
        return model.to(device).eval()

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.encoder.half().save_pretrained(path, safe_serialization=True)
        self.encoder.float()
        self.tokenizer.save_pretrained(path)
        torch.save(self.head.state_dict(), os.path.join(path, "sense_head.pt"))
        with open(os.path.join(path, "sense_labels.json"), "w", encoding="utf-8") as f:
            json.dump({"labels": self.labels, "word_labels": self.word_labels, "window": self.window,
                       "delimiters": self.delimiters}, f)

    def forward(self, occurrences: Sequence[Tuple[str, str, int, int]]) -> torch.Tensor:
        """Masked sense logits [N, n_labels] for (word, text, start, end) occurrences."""
        device = next(self.parameters()).device
        wins = [self._windower._window(t, int(a), int(b)) for _, t, a, b in occurrences]
        bt = self.tokenizer([w[0] for w in wins], return_offsets_mapping=True, padding=True,
                            truncation=True, max_length=96, return_tensors="pt")
        off = bt.pop("offset_mapping")
        h = self.encoder(**{k: v.to(device) for k, v in bt.items()}).last_hidden_state
        reps = []
        for j, (_, cs, ce) in enumerate(wins):
            sel = (off[j, :, 1] > off[j, :, 0]) & (off[j, :, 0] < ce) & (off[j, :, 1] > cs)
            if not bool(sel.any()):
                sel = off[j, :, 1] > off[j, :, 0]
            reps.append(h[j][sel.to(device)].mean(0))
        logits = self.head(self.drop(torch.stack(reps)))
        masks = torch.stack([self._masks.get(w.lower(), torch.ones(len(self.labels), dtype=torch.bool)) for w, *_ in occurrences])
        return logits.masked_fill(~masks.to(device), -1e4)

    def predict(self, occurrences: Sequence[Tuple[str, str, int, int]]) -> List[Dict[str, float]]:
        """{sense label: probability} per occurrence (only the word's own senses)."""
        if not occurrences:
            return []
        with torch.no_grad():
            probs = torch.softmax(self(occurrences).float(), dim=-1).cpu()
        out = []
        for p, (w, *_) in zip(probs, occurrences):
            out.append({l: float(p[self._index[l]]) for l in self.word_labels.get(w.lower(), [])})
        return out


_MODELS: Dict[str, Optional[SenseModel]] = {}
_LOCK = threading.Lock()


def get_sense_model(config) -> Optional[SenseModel]:
    """Shared fine-tuned sense model, or None when not trained/installed."""
    path = (getattr(config, "sense_model_path", "") or "").strip()
    if not path:
        return None
    with _LOCK:
        if path not in _MODELS:
            try:
                if not os.path.isfile(os.path.join(path, "sense_labels.json")):
                    raise FileNotFoundError(path)
                _MODELS[path] = SenseModel.load(path, getattr(config, "sense_encoder_device", "cpu"))
                logger.info(f"✓ Fine-tuned sense model ready: {path} ({len(_MODELS[path].word_labels)} homographs)")
            except Exception as e:
                _MODELS[path] = None
                logger.warning(f"Fine-tuned sense model not available ({e}); using the sense lexicon alone.")
        return _MODELS[path]
