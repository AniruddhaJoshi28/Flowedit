"""
GlossBERT word-sense disambiguation (Huang et al., EMNLP 2019; MIT licence).

Scores how well each WordNet definition of a word fits one occurrence: the occurrence is read
inside its clause-bounded window with the target word in quotes, paired with "word : definition".
The candidate definitions come from the sense lexicon file, so WordNet is not needed at runtime.
"""

import logging
import os
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from flowedit.memory.sense_encoder import ContextualSenseEncoder

logger = logging.getLogger(__name__)


class GlossWSD:
    def __init__(self, path: str, device: str, window_words: int, delimiters: str):
        from transformers import BertForSequenceClassification, BertTokenizer

        self.device = device
        self.tokenizer = BertTokenizer(os.path.join(path, "vocab.txt"), do_lower_case=True)
        model = BertForSequenceClassification.from_pretrained(path, num_labels=2).eval()
        self.model = (model.half() if device.startswith("cuda") else model).to(device)
        self._windower = ContextualSenseEncoder("", (0,), window_words, delimiters, cache_size=0)
        self._lock = threading.Lock()

    def sense_probs(self, text: str, start: int, end: int, glosses: Sequence[Tuple[str, str]]) -> Dict[str, float]:
        """{synset: P(definition fits this occurrence)} normalised over the word's senses."""
        if not glosses:
            return {}
        t, cs, ce = self._windower._window(text, start, end)
        context = t[:cs] + '" ' + t[cs:ce] + ' "' + t[ce:]
        with self._lock, torch.no_grad():
            enc = self.tokenizer([context] * len(glosses), [g for _, g in glosses], padding=True,
                                 truncation=True, max_length=128, return_tensors="pt").to(self.device)
            p = torch.softmax(self.model(**enc).logits.float(), dim=-1)[:, 1].cpu()
        z = float(p.sum()) or 1.0
        return {name: float(v) / z for (name, _), v in zip(glosses, p)}


_WSD: Dict[str, Optional[GlossWSD]] = {}
_LOCK = threading.Lock()


def get_gloss_wsd(config) -> Optional[GlossWSD]:
    """Shared GlossBERT model, or None when its weights are not installed."""
    path = (getattr(config, "sense_wsd_model_path", "") or "").strip()
    if not path:
        return None
    with _LOCK:
        if path not in _WSD:
            try:
                if not os.path.isfile(os.path.join(path, "vocab.txt")):
                    raise FileNotFoundError(path)
                device = getattr(config, "sense_wsd_device", "") or ("cuda" if torch.cuda.is_available() else "cpu")
                _WSD[path] = GlossWSD(path, device, int(config.sense_wsd_window_words), config.sense_clause_delimiters)
                logger.info(f"✓ GlossBERT word-sense model ready: {path} (device={device})")
            except Exception as e:
                _WSD[path] = None
                logger.warning(f"GlossBERT word-sense model not available ({e}); using the sense lexicon alone.")
        return _WSD[path]
