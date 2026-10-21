"""
Shared Homograph Context Resolver for FlowEdit.

Data-driven sense matching for registered spelling corrections: a candidate sense is described
only by the carrier sentences it was registered with, and an occurrence is assigned with the same
context sense model the Hopfield memory uses (flowedit/memory/sense_context.py).

There are no word lists, keyword tables, grammar rules or external models: pronunciations come
exclusively from user corrections (S3 spelling store / Hopfield memory).
"""

import re
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
    """Context-model sense matcher for registered pronunciation senses."""

    def __init__(
        self,
        window_radius: Optional[int] = None,
        config: Optional[MemoryConfig] = None,
    ):
        self.config = config or MemoryConfig()
        self.window_radius = window_radius or self.config.context_window

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
    # Sense matching (context sense model, flowedit/memory/sense_context.py)
    # ─────────────────────────────────────────────────────────────────────────

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

    def match_sense(
        self,
        text: str,
        word: str,
        candidate_senses: List[Dict[str, Any]],
        char_offset: Optional[int] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Select the registered sense matching this occurrence, or None (leave the word unedited).

        Registered senses are grouped by spelling; each group's carrier sentences are its examples
        in the context sense model. A single spelling applies only where the context supports it
        over the default reading; several spellings compete and the most probable wins when the
        context carries evidence. A global sense (no carrier context) applies when no contextual
        sense claims the word.
        """
        if not candidate_senses:
            return None
        from flowedit.memory import sense_context  # flowedit.memory imports this module via s3_storage

        global_cands = [c for c in candidate_senses if self._is_global(c, word)]
        contextual = [c for c in candidate_senses if not self._is_global(c, word)]

        q_span = self._occurrence_span(text, word, char_offset)
        if contextual and q_span is not None:
            window = int(self.config.sense_context_window)
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for c in contextual:
                groups.setdefault((c.get("spell_as") or "").strip().lower(), []).append(c)
            bg = sense_context.Background()
            senses = []
            for spelling, cands in groups.items():
                ctxs = []
                for c in cands:
                    carrier = (c.get("carrier_text") or "").strip()
                    bg.observe(carrier)
                    span = self._occurrence_span(carrier, word)
                    if span is not None:
                        ctxs.append(sense_context.context(carrier, span[0], window))
                senses.append(sense_context.SenseData(spelling, ctxs))
            bg.observe(text)
            keep = [i for i, s in enumerate(senses) if s.n > 0]
            if keep:
                kept = [senses[i] for i in keep]
                verdict, i, _ = sense_context.decide(
                    sense_context.context(text, q_span[0], window), kept,
                    sense_context.fit_lambda(kept, bg, window), bg,
                )
                if verdict == "choose":
                    return groups[kept[i].label][0]
                if verdict == "abstain":
                    return None

        return global_cands[0] if global_cands else None

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
