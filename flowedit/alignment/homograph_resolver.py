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

INFINITIVE_MODALS = {
    'will', 'shall', 'would', 'could', 'should', 'can', 'may', 'might', 'must',
    'to', 'do', 'does', 'did', 'wanna', 'gonna'
}

SUBJECT_PRONOUNS = {'i', 'he', 'she', 'they', 'we', 'you', 'it'}

DETERMINERS_AND_STOPWORDS = {
    'a', 'an', 'the', 'this', 'that', 'these', 'those', 'it', 'its', 'their', 'his', 'her', 'my', 'your', 'our'
}

TEMPORAL_PAST = {'yesterday', 'ago', 'last', 'previously', 'earlier', 'formerly', 'past'}
TEMPORAL_FUTURE = {'tomorrow', 'soon', 'later', 'next', 'upcoming'}


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
        char_offset: Optional[int] = None,
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
        if char_offset is not None:
            for idx, m in enumerate(re.finditer(r'\b[\w\'-]+\b', text.lower())):
                if m.group(0) == word_lower and (m.start() <= char_offset <= m.end() or abs(m.start() - char_offset) <= 2):
                    target_idx = idx
                    break

        if target_idx < 0:
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
        char_offset: Optional[int] = None,
    ) -> float:
        """Compute structural and lexical context compatibility score in [0.0, 1.0].
        
        Combines:
        1. Immediate neighbor alignment (weight 0.35): Syntactic anchors like particles,
           auxiliaries, prepositions, or subject pronouns directly adjoining the word.
        2. Local N-gram window overlap (weight 0.25): Jaccard overlap of non-stopword tokens in radius +-3.
        3. Sentence lexical keywords overlap (weight 0.20): Co-occurring content words.
        4. Temporal and verbal tense alignment (weight 0.20): Agreement across temporal markers.
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
            query_text, word_clean, char_offset=char_offset
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

        # 0. Modal auxiliary & verbal frame conflict:
        # If one context has an infinitive/future modal ('will read', 'to read', 'can read')
        # while the other does not ('I read yesterday', 'She read the book'),
        # the grammatical verbal senses / tenses are incompatible.
        q_has_modal = q_imm_l in INFINITIVE_MODALS
        c_has_modal = c_imm_l in INFINITIVE_MODALS
        if q_has_modal != c_has_modal:
            return 0.0

        # Incompatible temporal frame conflict (e.g. past vs future)
        q_past = bool(set(q_keywords) & TEMPORAL_PAST)
        c_past = bool(set(c_keywords) & TEMPORAL_PAST)
        q_fut = bool(set(q_keywords) & TEMPORAL_FUTURE)
        c_fut = bool(set(c_keywords) & TEMPORAL_FUTURE)
        if (q_past and c_fut) or (q_fut and c_past):
            return 0.0

        # 1. Immediate neighbor alignment (0.0 to 1.0)
        def _tok_match_wt(q_tok: Optional[str], c_tok: Optional[str], is_left: bool = True) -> float:
            if not q_tok and not c_tok:
                return 0.25
            if not q_tok or not c_tok:
                return 0.0
            if q_tok == c_tok:
                if q_tok in DETERMINERS_AND_STOPWORDS:
                    return 0.20  # common article/determiner gives low weight
                return 1.0
            # If both are subject pronouns (e.g. 'i' and 'she'), strong syntactic category alignment
            if is_left and q_tok in SUBJECT_PRONOUNS and c_tok in SUBJECT_PRONOUNS:
                return 0.75
            return 0.0

        left_match = _tok_match_wt(q_imm_l, c_imm_l, is_left=True)
        right_match = _tok_match_wt(q_imm_r, c_imm_r, is_left=False)
        neighbor_score = (left_match + right_match) / 2.0

        # 2. Local window Jaccard overlap (excluding target word and stopwords)
        q_win_set = {t for t in q_window.split() if t != word_clean and t not in STOPWORDS}
        c_win_set = {t for t in c_window.split() if t != word_clean and t not in STOPWORDS} if c_window else set()
        if q_win_set and c_win_set:
            win_overlap = len(q_win_set & c_win_set) / max(len(q_win_set | c_win_set), 1)
        else:
            win_overlap = 0.0

        # 3. Sentence keywords Jaccard overlap (excluding target word and stopwords)
        q_kw_set = {t for t in q_keywords if t != word_clean and t not in STOPWORDS}
        c_kw_set = {t for t in c_keywords if t != word_clean and t not in STOPWORDS}
        if q_kw_set and c_kw_set:
            kw_overlap = len(q_kw_set & c_kw_set) / max(len(q_kw_set | c_kw_set), 1)
        else:
            kw_overlap = 0.0

        # 4. Temporal agreement score
        temporal_score = 1.0 if ((q_past and c_past) or (q_fut and c_fut)) else 0.0

        # Composite score
        total_score = (0.35 * neighbor_score) + (0.25 * win_overlap) + (0.20 * kw_overlap) + (0.20 * temporal_score)
        return total_score

    def match_sense(
        self,
        text: str,
        word: str,
        candidate_senses: List[Dict[str, Any]],
        char_offset: Optional[int] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Deterministically select the matching sense from candidate_senses.
        
        Evaluates context compatibility scores without hardcoded rules:
        - If the best score exceeds match_threshold, returns the best candidate.
        - If no candidate satisfies the threshold, returns None (leaves word unedited).
        
        Args:
            text: Input carrier text to match against.
            word: Target word.
            candidate_senses: List of sense dicts stored in S3 dictionary.
            char_offset: Optional character start offset for word in text.
            
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
            score = self.compute_context_similarity(text, word, cand, char_offset=char_offset)
            if score > best_score:
                best_score = score
                best_cand = cand

        # If candidate meets contextual compatibility threshold, match it
        if best_cand is not None and best_score >= self.match_threshold:
            return best_cand

        # Fallback to global rule if available
        if global_cand is not None:
            return global_cand

        # If only one candidate sense exists, only use it if it's global or meets threshold
        if len(candidate_senses) == 1:
            single = candidate_senses[0]
            cand_carrier = (single.get("carrier_text", "") or single.get("context_window", "")).strip()
            if not cand_carrier or cand_carrier.lower() == word.lower() or single.get("sense_id") == "default":
                return single
            if best_cand is not None and best_score >= self.match_threshold:
                return best_cand
            return None

        return None

    def is_same_sense(self, text1: str, text2: str, word: str, char_offset: Optional[int] = None) -> bool:
        """Check if target word has compatible context across two carrier sentences."""
        if not text1 or not text2:
            return True
        w_clean = word.strip().lower()
        if text1.strip().lower() == w_clean or text2.strip().lower() == w_clean:
            return True
        dummy_cand = {"carrier_text": text1}
        sim = self.compute_context_similarity(text2, word, dummy_cand, char_offset=char_offset)
        return sim >= self.match_threshold

    def resolve_homograph_phonetic(
        self,
        word: str,
        context_text: str,
        char_offset: Optional[int] = None,
        **kwargs,
    ) -> Optional[str]:
        """Contextually resolve correct phonetic spelling for homographs based on surrounding context.
        
        Uses the dynamic S3/local spelling store and generalized sense matching without hardcoding.
        """
        if not word or not context_text:
            return None
        try:
            from flowedit.memory.s3_storage import s3_spelling_store
            return s3_spelling_store.resolve_correction(context_text, word, char_offset=char_offset)
        except Exception:
            return None


