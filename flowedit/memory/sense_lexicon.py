"""
Pretrained homograph sense lexicon with online learning from user corrections.

Built by scripts/build_sense_lexicon.py from a labelled corpus (Wikipedia Homograph Data). Every
homograph has a logistic-regression classifier over contextual occurrence embeddings; every sense
carries its pronunciation and a respelling the TTS reads with that pronunciation. No word lists
or rules live in code: coverage, senses and pronunciations all come from the data file.

User corrections teach it locally: a correction is stored as a labelled context example, and an
occurrence whose context (after removing the word's average feature) is close to a corrected
example follows that correction. Everything else keeps the corpus-trained decision, so a
correction fixes its own kind of sentence without changing unrelated ones.
"""

import logging
import math
import os
import re
import threading
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")
LOGIT_SCALE = 10.0  # features are unit-norm; scaling keeps the regularised optimum well-conditioned


def featurize(config, windows: Sequence[int], occurrences: Sequence[Tuple[str, int, int]], batch: int = 64) -> torch.Tensor:
    """[N, len(windows)·H] occurrence features: one clause-bounded encoding per window size."""
    from flowedit.memory.sense_encoder import get_sense_encoder

    parts = []
    for w in windows:
        enc = get_sense_encoder(config, context_words=int(w))
        if enc is None or not enc.available:
            raise RuntimeError("contextual sense encoder unavailable")
        rows = []
        for i in range(0, len(occurrences), batch):
            chunk = occurrences[i:i + batch]
            rows.append(enc.encode_batch(chunk))
        parts.append(torch.cat(rows) if rows else torch.zeros(0, 0))
    return torch.cat(parts, dim=1) / math.sqrt(len(windows))


def fit_classifier(x: torch.Tensor, y: torch.Tensor, n_classes: int, reg: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
    """L2-regularised multinomial logistic regression (LBFGS). Returns (W, b) with logits = x @ W + b."""
    x = x.float()
    if n_classes < 2 or len(torch.unique(y)) < 2:
        # Nothing to discriminate: constant prediction of the only observed class
        b = torch.full((n_classes,), -1e4)
        b[int(y[0]) if len(y) else 0] = 0.0
        return torch.zeros(x.shape[1], n_classes), b
    W = torch.zeros(x.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x @ W * LOGIT_SCALE + b, y) + reg * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach() * LOGIT_SCALE, b.detach()


class SenseLexicon:
    """Per-occurrence pronunciation sense prediction for every homograph in the lexicon file."""

    def __init__(self, config):
        self.config = config
        self.path = (config.sense_lexicon_path or "").strip()
        self.user_path = (config.sense_lexicon_user_path or "").strip()
        self._data = None
        self._failed = not self.path
        self._feature_config = None
        self._user: Dict[str, List[Tuple[torch.Tensor, str, str]]] = {}  # word -> [(centred feature, label, text)]
        self._lock = threading.RLock()

    # ── loading ──────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._load()

    def _load(self) -> bool:
        if self._data is not None:
            return True
        if self._failed:
            return False
        with self._lock:
            if self._data is not None:
                return True
            try:
                if not os.path.isfile(self.path):
                    raise FileNotFoundError(self.path)
                data = torch.load(self.path, map_location="cpu", weights_only=False)
                self._feature_config = replace(
                    self.config,
                    sense_encoder_model=data["encoder"],
                    sense_encoder_layers=tuple(data["layers"]),
                    sense_clause_delimiters=data.get("clause_delimiters", self.config.sense_clause_delimiters),
                )
                self._data = data
                self._load_user()
                if float(self.config.sense_model_weight) > 0:
                    from flowedit.memory.sense_model import get_sense_model
                    get_sense_model(self.config)  # warm the fine-tuned model alongside the lexicon
                if float(data.get("wsd_weight", 0.0)) > 0:
                    from flowedit.memory.gloss_wsd import get_gloss_wsd
                    get_gloss_wsd(self.config)  # warm the word-sense model alongside the lexicon
                logger.info(
                    f"✓ Sense lexicon ready: {len(data['words'])} homographs "
                    f"(held-out accuracy {data.get('heldout_accuracy', float('nan')):.3f}, "
                    f"{sum(len(v) for v in self._user.values())} user examples)"
                )
                return True
            except Exception as e:
                self._failed = True
                logger.warning(f"Sense lexicon unavailable ({e}); homographs rely on learned corrections only.")
                return False

    def _load_user(self) -> None:
        if not self.user_path or not os.path.isfile(self.user_path):
            return
        try:
            saved = torch.load(self.user_path, map_location="cpu", weights_only=False)
            for word, items in (saved.get("examples") or {}).items():
                if word in self._data["words"]:
                    self._user[word] = [(f, l, t) for f, l, t in items]
        except Exception as e:
            logger.warning(f"Could not load sense lexicon user examples from {self.user_path}: {e}")

    def _save_user(self) -> None:
        if not self.user_path:
            return
        try:
            torch.save({"examples": self._user}, self.user_path)
        except Exception as e:
            logger.warning(f"Could not save sense lexicon user examples: {e}")

    # ── queries ──────────────────────────────────────────────────────────────

    def covers(self, word: str) -> bool:
        return self._load() and word.strip().lower() in self._data["words"]

    def occurrences(self, text: str) -> List[Tuple[str, int, int]]:
        """(word, start, end) of every lexicon homograph in `text`."""
        if not self._load() or not text:
            return []
        words = self._data["words"]
        return [(m.group(0).lower(), m.start(), m.end()) for m in _WORD_RE.finditer(text) if m.group(0).lower() in words]

    def labels(self, word: str) -> List[str]:
        return list(self._data["words"][word.lower()]["labels"]) if self.covers(word) else []

    def respelling(self, word: str, label: str) -> Optional[str]:
        return self._data["words"][word.lower()]["respelling"].get(label) if self.covers(word) else None

    def sense_counts(self, word: str) -> Dict[str, int]:
        return dict(self._data["words"][word.lower()].get("counts") or {}) if self.covers(word) else {}

    def phones(self, word: str, label: str) -> Optional[str]:
        return self._data["words"][word.lower()]["phones"].get(label) if self.covers(word) else None

    def _centred(self, word: str, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x - self._data["words"][word]["mean"], p=2, dim=-1)

    def features(self, text: str, spans: Sequence[Tuple[int, int]]) -> torch.Tensor:
        return featurize(self._feature_config, self._data["windows"], [(text, a, b) for a, b in spans])

    def predict(self, text: str, occurrences: Sequence[Tuple[str, int, int]]) -> List[Tuple[str, float]]:
        """(label, probability) for each (word, start, end) occurrence, in order."""
        if not occurrences or not self._load():
            return []
        x = self.features(text, [(a, b) for _, a, b in occurrences])
        model_probs = self._model_probs(text, occurrences)
        weight = float(self.config.sense_model_weight)
        out = []
        radius = float(self.config.sense_lexicon_user_radius)
        for i, (word, _, _) in enumerate(occurrences):
            d = self._data["words"][word]
            p = torch.softmax(x[i] @ d["W"] + d["b"], dim=-1)
            mp = model_probs[i] if model_probs else None
            if mp and set(mp) == set(d["labels"]):
                p = (1.0 - weight) * p + weight * torch.tensor([mp[l] for l in d["labels"]])
            if float(p.max()) < float(self.config.sense_wsd_skip_confidence):
                pw = self._wsd_probs(word, text, occurrences[i][1], occurrences[i][2])
                if pw is not None:
                    w_wsd = float(self._data.get("wsd_weight", 0.0))
                    p = torch.softmax(torch.log(p + 1e-9) + w_wsd * torch.log(pw + 1e-9), dim=-1)
            k = int(p.argmax())
            label, prob = d["labels"][k], float(p[k])
            taught = self._user.get(word)
            if taught:
                sims = torch.stack([f for f, _, _ in taught]) @ self._centred(word, x[i])
                j = int(sims.argmax())
                if float(sims[j]) >= radius:
                    label, prob = taught[j][1], max(prob if taught[j][1] == label else 0.0, float(sims[j]))
            out.append((label, prob))
        return out

    def _model_probs(self, text: str, occurrences) -> Optional[List[Dict[str, float]]]:
        """Fine-tuned sense model probabilities (None when not installed or disabled)."""
        if float(self.config.sense_model_weight) <= 0:
            return None
        from flowedit.memory.sense_model import get_sense_model
        model = get_sense_model(self.config)
        if model is None:
            return None
        try:
            return model.predict([(w, text, a, b) for w, a, b in occurrences])
        except Exception as e:
            logger.warning(f"Fine-tuned sense model failed on {text[:60]!r}: {e}")
            return None

    def _wsd_probs(self, word: str, text: str, start: int, end: int) -> Optional[torch.Tensor]:
        """P(pronunciation | WordNet sense of this occurrence), learned sense→pronunciation table."""
        d = self._data["words"][word]
        glosses, table = d.get("wsd_glosses"), d.get("wsd_table")
        if not glosses or not table or not float(self._data.get("wsd_weight", 0.0)):
            return None
        from flowedit.memory.gloss_wsd import get_gloss_wsd
        wsd = get_gloss_wsd(self.config)
        if wsd is None:
            return None
        try:
            senses = wsd.sense_probs(text, start, end, glosses)
        except Exception as e:
            logger.warning(f"GlossBERT failed on {text[:60]!r}: {e}")
            return None
        labels = d["labels"]
        counts = d.get("counts") or {}
        total = sum(counts.values()) or 1
        prior = torch.tensor([(counts.get(l, 0) + 1) / (total + len(labels)) for l in labels])
        alpha = float(self.config.sense_wsd_smoothing)
        out = torch.zeros(len(labels))
        for synset, ps in senses.items():
            row = table.get(synset, {})
            n = sum(row.values())
            out += ps * (torch.tensor([row.get(l, 0.0) for l in labels]) + alpha * prior) / (n + alpha)
        return out / out.sum()

    # ── learning from corrections ────────────────────────────────────────────

    def label_for_spelling(self, word: str, spelling: Optional[str]) -> Optional[str]:
        """Sense whose pronunciation the user's spelling denotes, or None if it cannot be told.

        Exact match with a sense respelling first; otherwise, when a G2P model is installed
        (g2p_en), the sense whose phones are strictly closest to the spelling's phones.
        """
        if not spelling or not self.covers(word):
            return None
        w = word.lower()
        s = re.sub(r"[^a-z]", "", spelling.lower())
        if not s or s == w:
            return None
        d = self._data["words"][w]
        for label, r in d["respelling"].items():
            if r and r.lower() == s:
                return label
        g2p = _g2p()
        if g2p is None:
            return None
        guess = [re.sub(r"\d", "", p) for p in g2p(spelling) if re.fullmatch(r"[A-Z]+\d?", p)]
        scored = []
        for label, ph in d["phones"].items():
            if ph:
                scored.append((_edit_distance(guess, ph.split()), label))
        scored.sort()
        if len(scored) >= 2 and scored[0][0] < scored[1][0]:
            return scored[0][1]
        return None

    def learn(self, word: str, text: str, span: Tuple[int, int], label: str) -> None:
        """Store a user-confirmed (context → sense) example; similar contexts will follow it."""
        w = word.lower()
        if not self.covers(w) or label not in self._data["words"][w]["labels"]:
            return
        with self._lock:
            feat = self._centred(w, self.features(text, [span])[0])
            items = [it for it in self._user.get(w, []) if not (it[2] == text and torch.allclose(it[0], feat))]
            items.append((feat, label, text))
            self._user[w] = items
            self._save_user()
        logger.info(f"✓ Sense lexicon learned '{w}' → {label} from: {text!r} ({len(self._user[w])} user examples)")


_G2P = None


def _g2p():
    """Optional g2p_en model (used only to map free-form user spellings to a sense)."""
    global _G2P
    if _G2P is None:
        try:
            from g2p_en import G2p
            _G2P = G2p()
        except Exception:
            _G2P = False
    return _G2P or None


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b) + 1):
            cur = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, dp[j] = dp[j], cur
    return dp[-1]


_LEXICONS: Dict[Tuple[str, str], SenseLexicon] = {}
_LEXICONS_LOCK = threading.Lock()


def get_sense_lexicon(config) -> Optional[SenseLexicon]:
    """Process-wide shared lexicon (XTTS and F5-TTS see the same senses and user examples)."""
    path = (getattr(config, "sense_lexicon_path", "") or "").strip()
    if not path:
        return None
    key = (os.path.abspath(path), os.path.abspath(config.sense_lexicon_user_path or ""))
    with _LEXICONS_LOCK:
        lex = _LEXICONS.get(key)
        if lex is None:
            lex = SenseLexicon(config)
            _LEXICONS[key] = lex
    return lex if lex.available else None
