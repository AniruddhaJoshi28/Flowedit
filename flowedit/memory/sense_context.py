"""
Context sense model for homographs, learned only from corrections and from text the memory has seen.

No external model, word list or grammar rule: every number below is estimated from the memory's
own data, and every pronunciation comes from a user correction.

Model
-----
An occurrence x of word w has context features  φ(x) = {(d, t_d) : d = ±1 … ±W},  where t_d is the
lower-cased token at offset d (sentence edges are the tokens "<s>" and "</s>").

  Background      p_bg(t) = (c(t) + 1) / (N + V + 1)           unigram over all text the memory has seen
  Sense s         q_s(d, t) = n_s(d, t) / |C_s|                 share of the corrections of s with t at d
  Feature         p(t_d | s)  = λ_d q_s(d, t_d) + (1 − λ_d) p_bg(t_d)
  Default         p(t_d | s0) = p_bg(t_d)                       the TTS model's own reading
  Posterior       log P(s | x) = log π_s + Σ_d log p(t_d | s) + const,   π_s ∝ |C_s|,  π_0 ∝ 1

λ_d is how much offset d carries pronunciation evidence. It is fitted by EM on a two-component
mixture with leave-one-out sense estimates over every correction in memory (all words pooled, so
corrections of one homograph teach which positions matter for the next), with a uniform Beta(1, 1)
prior: no corrections → λ_d = 1/2; corrections whose contexts never recur within a sense → λ_d → 0
and context stops influencing the choice.

Decision
--------
  one learned pronunciation   apply it iff P(s | x) > P(s0 | x), else leave the word to the TTS model
  several pronunciations      the corrections cover the word's readings, s0 is dropped: argmax_s
                              P(s | x) when some context token of x was seen with one of them,
                              otherwise undecided (the caller keeps its key-similarity choice)
"""

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_TOKEN = re.compile(r"[\w']+|[^\w\s]")
BOS, EOS = "<s>", "</s>"


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def context(text: str, char_start: int, window: int) -> List[Tuple[int, str]]:
    """[(offset, token)] around the word that starts at `char_start`."""
    spans = [(m.group().lower(), m.start()) for m in _TOKEN.finditer(text or "")]
    idx = next((i for i, (_, s) in enumerate(spans) if s >= char_start), len(spans))
    toks = [t for t, _ in spans]
    feats = []
    for d in range(1, window + 1):
        j = idx - d
        feats.append((-d, toks[j] if j >= 0 else BOS))
        j = idx + d
        feats.append((d, toks[j] if j < len(toks) else EOS))
    return feats


class Background:
    """Unigram token statistics of every sentence the memory has seen (unsupervised, incremental)."""

    def __init__(self, counts: Optional[Dict[str, int]] = None):
        self.counts: Counter = Counter(counts or {})
        self.total = sum(self.counts.values())

    def observe(self, text: str) -> None:
        toks = tokenize(text) + [BOS, EOS]
        self.counts.update(toks)
        self.total += len(toks)

    def prob(self, token: str) -> float:
        return (self.counts.get(token, 0) + 1.0) / (self.total + len(self.counts) + 1.0)


class SenseData:
    """Correction contexts of one pronunciation of one word."""

    def __init__(self, label: str, contexts: Sequence[List[Tuple[int, str]]]):
        self.label = label
        self.contexts = list(contexts)
        self.n = len(self.contexts)
        self.counts: Counter = Counter(f for c in self.contexts for f in c)

    def q(self, feature: Tuple[int, str], exclude: Optional[List[Tuple[int, str]]] = None) -> float:
        n, c = self.n, self.counts.get(feature, 0)
        if exclude is not None:
            n -= 1
            c -= exclude.count(feature)
        return c / n if n > 0 else 0.0


def fit_lambda(senses: Iterable[SenseData], bg: Background, window: int, iters: int = 100) -> Dict[int, float]:
    """λ_d by EM with leave-one-out sense estimates and a Beta(1, 1) prior."""
    points: Dict[int, List[Tuple[float, float]]] = {d: [] for w in range(1, window + 1) for d in (-w, w)}
    for s in senses:
        if s.n < 2:
            continue  # leave-one-out needs another correction of the same pronunciation
        for ctx in s.contexts:
            for f in ctx:
                points[f[0]].append((s.q(f, exclude=ctx), bg.prob(f[1])))
    lam = {}
    for d, pts in points.items():
        l = 0.5
        for _ in range(iters):
            r = sum(l * q / (l * q + (1 - l) * p) for q, p in pts)
            l_new = (r + 1.0) / (len(pts) + 2.0)
            if abs(l_new - l) < 1e-6:
                l = l_new
                break
            l = l_new
        lam[d] = l
    return lam


def decide(
    occurrence: List[Tuple[int, str]],
    senses: Sequence[SenseData],
    lam: Dict[int, float],
    bg: Background,
) -> Tuple[str, Optional[int], Dict]:
    """('choose', i, info) | ('abstain', None, info) | ('undecided', None, info)."""
    logl = []
    for s in senses:
        ll = math.log(s.n)
        for f in occurrence:
            ll += math.log(lam[f[0]] * s.q(f) + (1 - lam[f[0]]) * bg.prob(f[1]))
        logl.append(ll)
    default = sum(math.log(bg.prob(t)) for _, t in occurrence)  # π_0 ∝ 1 → log 1 = 0
    info: Dict = {"labels": [s.label for s in senses], "log_lik": logl, "log_lik_default": default,
                  "matched": [sorted(f for f in occurrence if s.counts.get(f)) for s in senses]}
    z = max(logl + [default]) if len(senses) == 1 else max(logl)
    terms = logl + ([default] if len(senses) == 1 else [])
    norm = z + math.log(sum(math.exp(v - z) for v in terms))
    info["posterior"] = [math.exp(v - norm) for v in terms]
    best = max(range(len(senses)), key=lambda i: logl[i])
    if len(senses) == 1:
        return ("choose", 0, info) if logl[0] > default else ("abstain", None, info)
    if not any(info["matched"]):
        return "undecided", None, info
    return "choose", best, info
