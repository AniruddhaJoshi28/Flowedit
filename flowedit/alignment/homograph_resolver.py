"""
Shared Homograph Context Resolver for FlowEdit.

Data-driven sense matching for registered spelling corrections: a candidate sense is described
only by the carrier sentence it was registered with. Compatibility between an occurrence and a
sense is the cosine similarity of their contextual occurrence embeddings (the same
backbone-independent encoder the Hopfield memory uses for its EMA sense prototypes).

There are no word lists, keyword tables or grammar rules: pronunciations come exclusively from
user corrections (S3 spelling store / Hopfield memory). When the contextual encoder is
unavailable, a distance-weighted lexical overlap of the surrounding words is used instead.
"""

import re
import math
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from flowedit.config import MemoryConfig

_TOKEN_RE = re.compile(r"\b[\w'-]+\b")


@dataclass
class SenseProfile:
    """Represents a generalized context sense fingerprint for a target word in a sentence."""
    word: str
    sense_id: str
    display_name: str
    confidence: float
    context_window: str
    left_tokens: List[str] = field(default_factory=list)
    right_tokens: List[str] = field(default_factory=list)
    immediate_left: str = ""
    immediate_right: str = ""
    keywords: List[str] = field(default_factory=list)


class HomographContextResolver:
    """Contextual-embedding sense matcher for registered pronunciation senses."""

    def __init__(
        self,
        window_radius: Optional[int] = None,
        match_threshold: Optional[float] = None,
        config: Optional[MemoryConfig] = None,
        encoder: Any = None,
    ):
        self.config = config or MemoryConfig()
        self.window_radius = window_radius or self.config.context_window
        self.match_threshold = self.config.sense_match_threshold if match_threshold is None else match_threshold
        self.min_similarity = self.config.sense_min_similarity
        self.lexical_threshold = self.config.sense_lexical_threshold
        self._encoder = encoder

    # ─────────────────────────────────────────────────────────────────────────
    # Context features (structural metadata stored with spelling senses)
    # ─────────────────────────────────────────────────────────────────────────

    def extract_context_features(
        self,
        text: str,
        word: str,
        radius: Optional[int] = None,
        char_offset: Optional[int] = None,
    ) -> Tuple[List[str], List[str], str, str, str, List[str]]:
        """Extract structural and lexical context features around the target word occurrence.

        Returns:
            Tuple of (left_tokens, right_tokens, immediate_left, immediate_right, window_str, keywords)
        """
        r = radius or self.window_radius
        matches = list(_TOKEN_RE.finditer(text.lower()))
        tokens = [m.group(0) for m in matches]
        word_lower = word.strip().lower()

        target_idx = self._target_token_index(matches, word_lower, char_offset)

        if target_idx >= 0:
            start_idx = max(0, target_idx - r)
            end_idx = min(len(tokens), target_idx + r + 1)
            left_tokens = tokens[start_idx:target_idx]
            right_tokens = tokens[target_idx + 1:end_idx]
            window_tokens = tokens[start_idx:end_idx]
        else:
            left_tokens = []
            right_tokens = []
            window_tokens = tokens[:2 * r + 1]

        immediate_left = left_tokens[-1] if left_tokens else ""
        immediate_right = right_tokens[0] if right_tokens else ""
        window_str = " ".join(window_tokens)
        keywords = list(dict.fromkeys(t for t in tokens if len(t) > 1 and t != word_lower))

        return left_tokens, right_tokens, immediate_left, immediate_right, window_str, keywords

    @staticmethod
    def _target_token_index(matches, word_lower: str, char_offset: Optional[int]) -> int:
        hits = [i for i, m in enumerate(matches) if m.group(0) == word_lower]
        if not hits:
            return -1
        if char_offset is None:
            return hits[0]
        return min(hits, key=lambda i: abs(matches[i].start() - char_offset))

    def classify_sense(self, text: str, word: str) -> SenseProfile:
        """Derive a deterministic structural sense profile (id + display metadata) for word in text."""
        word_clean = word.strip().lower()
        if not text.strip():
            return SenseProfile(
                word=word_clean,
                sense_id="default",
                display_name="Global / Default",
                confidence=1.0,
                context_window="",
            )

        left_tokens, right_tokens, imm_left, imm_right, window_str, keywords = self.extract_context_features(
            text, word_clean
        )

        parts = []
        if imm_left:
            parts.append(f"L:{imm_left}")
        parts.append(f"[{word_clean}]")
        if imm_right:
            parts.append(f"R:{imm_right}")

        # Deterministic sense hash from structural window
        anchor_str = f"{imm_left}_{word_clean}_{imm_right}"
        hash_suffix = hashlib.md5(" ".join(left_tokens + [word_clean] + right_tokens).encode("utf-8")).hexdigest()[:6]
        sense_id = f"ctx_{anchor_str}_{hash_suffix}" if (imm_left or imm_right) else "default"

        display_name = f"Context: {' '.join(parts)}" if parts else f"Context: [{word_clean}]"

        return SenseProfile(
            word=word_clean,
            sense_id=sense_id,
            display_name=display_name,
            confidence=1.0 if (imm_left or imm_right) else 0.5,
            context_window=window_str,
            left_tokens=left_tokens,
            right_tokens=right_tokens,
            immediate_left=imm_left,
            immediate_right=imm_right,
            keywords=keywords,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Similarity
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def encoder(self):
        if self._encoder is None:
            from flowedit.memory.sense_encoder import get_sense_encoder
            self._encoder = get_sense_encoder(self.config) or False
        if not self._encoder or not self._encoder.available:
            return None
        return self._encoder

    @staticmethod
    def _occurrence_span(text: str, word: str, char_offset: Optional[int] = None) -> Optional[Tuple[int, int]]:
        spans = [m.span() for m in re.finditer(r"\b" + re.escape(word.strip()) + r"\b", text, flags=re.IGNORECASE)]
        if not spans:
            return None
        if char_offset is None:
            return spans[0]
        return min(spans, key=lambda s: abs(s[0] - char_offset))

    def _is_global(self, cand: Dict[str, Any], word: str) -> bool:
        carrier = (cand.get("carrier_text") or "").strip().strip(".!?,;:").strip().lower()
        return not carrier or carrier == word.strip().lower() or cand.get("sense_id") == "default"

    def _lexical_similarity(self, query_text: str, word: str, carrier: str, char_offset: Optional[int]) -> float:
        """Distance-weighted overlap of the words surrounding each occurrence (encoder fallback)."""
        w = word.strip().lower()

        def weights(text: str, offset: Optional[int]) -> Dict[str, float]:
            matches = list(_TOKEN_RE.finditer(text.lower()))
            idx = self._target_token_index(matches, w, offset)
            if idx < 0:
                return {}
            out: Dict[str, float] = {}
            sigma = float(self.window_radius)
            for j, m in enumerate(matches):
                if j == idx or m.group(0) == w:
                    continue
                wt = math.exp(-((j - idx) ** 2) / (2.0 * sigma * sigma))
                out[m.group(0)] = max(out.get(m.group(0), 0.0), wt)
            return out

        q, c = weights(query_text, char_offset), weights(carrier, None)
        if not q or not c:
            return 0.0
        inter = sum(min(q[t], c[t]) for t in q.keys() & c.keys())
        union = sum(max(q.get(t, 0.0), c.get(t, 0.0)) for t in q.keys() | c.keys())
        return inter / union if union > 0 else 0.0

    def compute_context_similarity(
        self,
        query_text: str,
        word: str,
        candidate_entry: Dict[str, Any],
        char_offset: Optional[int] = None,
    ) -> float:
        """Context compatibility in [0, 1] between a word occurrence and a registered sense.

        Global senses (registered without sentence context) are compatible with every context.
        """
        if self._is_global(candidate_entry, word):
            return 1.0
        carrier = (candidate_entry.get("carrier_text") or "").strip()
        enc = self.encoder
        q_span = self._occurrence_span(query_text, word, char_offset)
        c_span = self._occurrence_span(carrier, word)
        if enc is not None and q_span is not None and c_span is not None:
            vq = enc.encode(query_text, [q_span])
            vc = enc.encode(carrier, [c_span])
            if vq is not None and vc is not None:
                return float((vq[0] * vc[0]).sum())
        return self._lexical_similarity(query_text, word, carrier, char_offset)

    def _thresholds(self) -> Tuple[float, float]:
        if self.encoder is None:
            return self.lexical_threshold, 0.0
        return self.match_threshold, self.min_similarity

    def match_sense(
        self,
        text: str,
        word: str,
        candidate_senses: List[Dict[str, Any]],
        char_offset: Optional[int] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Select the registered sense matching this occurrence, or None (leave the word unedited).

        - Several contextual senses with different spellings: the most similar sense wins if it
          reaches the multi-sense floor (relative decision).
        - A single contextual sense: it must reach the single-sense threshold.
        - A global sense (no carrier context) applies when no contextual sense claims the word.
        """
        if not candidate_senses:
            return None

        global_cands = [c for c in candidate_senses if self._is_global(c, word)]
        contextual = [c for c in candidate_senses if not self._is_global(c, word)]

        if contextual:
            single_thr, multi_thr = self._thresholds()
            scores = [self.compute_context_similarity(text, word, c, char_offset=char_offset) for c in contextual]
            spellings = {(c.get("spell_as") or "").strip().lower() for c in contextual}
            threshold = multi_thr if len(spellings) > 1 else single_thr
            j = max(range(len(scores)), key=lambda k: scores[k])
            if scores[j] >= threshold and (len(spellings) > 1 or scores[j] > 0.0):
                return contextual[j]

        return global_cands[0] if global_cands else None

    def is_same_sense(self, text1: str, text2: str, word: str, char_offset: Optional[int] = None) -> bool:
        """Check if target word has compatible context across two carrier sentences."""
        if not text1 or not text2:
            return True
        w_clean = word.strip().lower()
        if text1.strip().lower() == w_clean or text2.strip().lower() == w_clean:
            return True
        single_thr, _ = self._thresholds()
        sim = self.compute_context_similarity(text2, word, {"carrier_text": text1}, char_offset=char_offset)
        return sim >= single_thr

    def resolve_homograph_phonetic(
        self,
        word: str,
        context_text: str,
        char_offset: Optional[int] = None,
        **kwargs,
    ) -> Optional[str]:
        """Registered pronunciation for this occurrence of `word`, or None if none matches its context."""
        if not word or not context_text:
            return None
        try:
            from flowedit.memory.s3_storage import s3_spelling_store
            spelled = s3_spelling_store.resolve_correction(context_text, word, char_offset=char_offset)
        except Exception:
            return None
        if spelled and spelled.strip().lower() != word.strip().lower():
            return spelled.strip()
        return None
