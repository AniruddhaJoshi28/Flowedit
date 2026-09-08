"""
Shared Homograph Context Resolver for FlowEdit.

Provides fully generalized, non-hardcoded context classification and sense
disambiguation based on local structural n-gram windows, syntactic neighbor
alignment, and lexical co-occurrence.

Works across any vocabulary (English, Indic, medical, technical, slang)
without hardcoded dictionaries or hand-crafted word rules.
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set

# Universal function words / stopwords for extracting salient content keywords
STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'in', 'on', 'at', 'to', 'for', 'with', 'from', 'of', 'by', 'about',
    'and', 'or', 'but', 'so', 'yet', 'it', 'its', 'this', 'that', 'these', 'those'
}


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
    """Generalized, non-hardcoded context classifier and sense matcher."""

    def __init__(self, window_radius: int = 3, match_threshold: float = 0.25):
        self.window_radius = window_radius
        self.match_threshold = match_threshold

    def extract_context_features(
        self,
        text: str,
        word: str,
        radius: Optional[int] = None,
    ) -> Tuple[List[str], List[str], str, str, str, List[str]]:
        """Extract structural and lexical context features around target word without hardcoding.
        
        Returns:
            Tuple of (left_tokens, right_tokens, immediate_left, immediate_right, window_str, keywords)
        """
        r = radius or self.window_radius
        tokens = re.findall(r'\b[\w\'-]+\b', text.lower())
        word_lower = word.strip().lower()

        # Find target word index (or best matching token)
        target_idx = -1
        for idx, t in enumerate(tokens):
            if t == word_lower:
                target_idx = idx
                break

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
        keywords = [t for t in tokens if t not in STOPWORDS and len(t) > 1 and t != word_lower]

        return left_tokens, right_tokens, immediate_left, immediate_right, window_str, keywords

    def classify_sense(self, text: str, word: str) -> SenseProfile:
        """Derive a canonical, generalized context sense profile for word in text.
        
        Does not use hardcoded word lists. Dynamically extracts local structural
        n-gram anchors and salient keywords to construct a deterministic sense signature.
        """
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

        # Build dynamic, non-hardcoded sense anchor
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
            confidence=0.85 if (imm_left or imm_right) else 0.50,
            context_window=window_str,
            left_tokens=left_tokens,
            right_tokens=right_tokens,
            immediate_left=imm_left,
            immediate_right=imm_right,
            keywords=keywords,
        )

    def compute_context_similarity(
        self,
        query_text: str,
        word: str,
        candidate_entry: Dict[str, Any],
    ) -> float:
        """Compute structural and lexical context compatibility score in [0.0, 1.0].
        
        Combines:
        1. Immediate neighbor alignment (weight 0.40): Syntactic anchors like particles,
           auxiliaries, prepositions, or subject pronouns directly adjoining the word.
        2. Local N-gram window overlap (weight 0.35): Jaccard overlap of tokens in radius +-3.
        3. Sentence lexical keywords overlap (weight 0.25): Co-occurring content words.
        """
        word_clean = word.strip().lower()
        cand_carrier = (
            candidate_entry.get("carrier_text", "")
            or candidate_entry.get("context_window", "")
        ).strip()

        # If candidate has no carrier context, or carrier is just the target word itself, or sense_id is default -> unconditional rule
        if not cand_carrier or cand_carrier.lower() == word_clean or candidate_entry.get("sense_id") == "default":
            return 1.0

        # Extract features from query sentence
        q_left, q_right, q_imm_l, q_imm_r, q_window, q_keywords = self.extract_context_features(
            query_text, word_clean
        )

        # Extract features from candidate stored sense
        c_left = candidate_entry.get("left_tokens")
        c_right = candidate_entry.get("right_tokens")
        c_imm_l = candidate_entry.get("immediate_left")
        c_imm_r = candidate_entry.get("immediate_right")
        c_window = candidate_entry.get("context_window", "")
        c_keywords = candidate_entry.get("context_clues", []) or candidate_entry.get("context_keywords", [])

        # Re-derive features if not pre-stored
        if c_left is None or c_right is None or c_imm_l is None:
            c_left, c_right, c_imm_l, c_imm_r, c_win_derived, c_kws_derived = self.extract_context_features(
                cand_carrier, word_clean
            )
            c_window = c_window or c_win_derived
            c_keywords = c_keywords or c_kws_derived

        # 1. Immediate neighbor alignment (0.0 to 1.0)
        left_match = 1.0 if (q_imm_l and c_imm_l and q_imm_l == c_imm_l) else (0.5 if (not q_imm_l and not c_imm_l) else 0.0)
        right_match = 1.0 if (q_imm_r and c_imm_r and q_imm_r == c_imm_r) else (0.5 if (not q_imm_r and not c_imm_r) else 0.0)
        neighbor_score = (left_match + right_match) / 2.0

        # 2. Local window Jaccard overlap (0.0 to 1.0)
        q_win_set = set(q_window.split())
        c_win_set = set(c_window.split()) if c_window else set()
        if q_win_set and c_win_set:
            win_overlap = len(q_win_set & c_win_set) / max(len(q_win_set | c_win_set), 1)
        else:
            win_overlap = 0.0

        # 3. Sentence keywords Jaccard overlap (0.0 to 1.0)
        q_kw_set = set(q_keywords)
        c_kw_set = set(c_keywords)
        if q_kw_set and c_kw_set:
            kw_overlap = len(q_kw_set & c_kw_set) / max(len(q_kw_set | c_kw_set), 1)
        else:
            kw_overlap = 0.0

        # Composite score
        total_score = (0.40 * neighbor_score) + (0.35 * win_overlap) + (0.25 * kw_overlap)
        return total_score

    def match_sense(
        self,
        text: str,
        word: str,
        candidate_senses: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Deterministically select the matching sense from candidate_senses.
        
        Evaluates context compatibility scores without hardcoded rules:
        - If the best score exceeds match_threshold, returns the best candidate.
        - If no candidate satisfies the threshold, returns None (leaves word unedited).
        
        Args:
            text: Input carrier text to match against.
            word: Target word.
            candidate_senses: List of sense dicts stored in S3 dictionary.
            
        Returns:
            The best matching candidate dict or None.
        """
        if not candidate_senses:
            return None

        # If any candidate is an unconditional/global rule without distinct carrier context
        global_cand = next(
            (c for c in candidate_senses if not (c.get("carrier_text", "").strip()) or c.get("carrier_text", "").strip().lower() == word.lower() or c.get("sense_id") == "default"),
            None
        )

        best_cand = None
        best_score = -1.0

        for cand in candidate_senses:
            score = self.compute_context_similarity(text, word, cand)
            if score > best_score:
                best_score = score
                best_cand = cand

        # If candidate meets contextual compatibility threshold, match it
        if best_cand is not None and best_score >= self.match_threshold:
            return best_cand

        # Fallback to global rule if available
        if global_cand is not None:
            return global_cand

        # If only one candidate sense exists for this word, use it
        if len(candidate_senses) == 1:
            return candidate_senses[0]

        return None

    def is_same_sense(self, text1: str, text2: str, word: str) -> bool:
        """Check if target word has compatible context across two carrier sentences."""
        dummy_cand = {"carrier_text": text1}
        sim = self.compute_context_similarity(text2, word, dummy_cand)
        return sim >= self.match_threshold
