"""
Grammatical-frame sense memory for homographs (no external model).

Content words around a homograph vary without limit, but the closed-class words that frame it
("to ___", "the ___", "of ___", "she ___ the") form a small, fixed set, and English homograph
readings mostly differ by grammatical role (noun vs verb, base form vs past tense, mass vs count).

Every stored correction is an example "<word> in frame F is pronounced P". A new occurrence is
described by the same frame features and follows the corrections that share them, independent of
its content words. Pronunciations come only from the user's corrections; the word classes below
describe English grammar and never name a pronunciation.

Decision for one occurrence of a word with learned senses:
  - features in back-off order: (left, right) frame, then left only, then right only;
  - the first level where the learned senses disagree in evidence decides (majority, strict);
  - single learned pronunciation: it applies only where the left frame (or the full frame) matches
    one of its corrections; anywhere else memory abstains and the TTS model reads the word, so a
    correction is never forced onto a grammatical role it was not learned in;
  - otherwise undecided: the caller keeps its existing selection.
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

DET = {"the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our",
       "their", "some", "any", "no", "every", "each", "another", "which", "what", "whose", "one"}
TO = {"to"}
MODAL = {"will", "would", "can", "could", "shall", "should", "may", "might", "must", "do", "does",
         "did", "let", "lets", "'ll", "won't", "can't", "don't", "doesn't", "didn't", "please"}
PERF = {"have", "has", "had", "having", "'ve", "'d"}
BE = {"is", "are", "was", "were", "be", "been", "being", "am", "'s", "'re", "'m"}
PRON = {"i", "you", "he", "she", "we", "they", "it", "who"}
PRON_OBJ = {"me", "him", "us", "them", "it", "her", "you"}
PREP = {"of", "in", "on", "at", "from", "with", "by", "for", "about", "into", "through", "without",
        "under", "over", "after", "before", "during", "against", "between", "as", "like", "toward",
        "towards", "around", "across", "behind", "near", "onto", "upon", "within"}
CONJ = {"and", "or", "but", "nor", "so", "yet"}
# Closed-class words skipped when looking for the frame word on the left ("has already read",
# "will not lead", "she quickly wound")
SKIP_LEFT = {"not", "never", "already", "just", "also", "still", "always", "often", "then", "even",
             "only", "really", "usually", "finally", "first", "again", "n't"}

NOUN_LIKE = {"DET", "DET_MOD", "PREP"}
VERB_LIKE = {"TO", "MODAL", "PERF", "PRON", "BE"}

_TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,;:!?\"()]")


def _cls(tok: Optional[str]) -> str:
    if tok is None or not tok[0].isalpha():
        return "START"
    t = tok.lower()
    for name, words in (("DET", DET), ("TO", TO), ("MODAL", MODAL), ("PERF", PERF), ("BE", BE),
                        ("PRON", PRON), ("PREP", PREP), ("CONJ", CONJ)):
        if t in words:
            return name
    return "OPEN"


def _right_cls(tok: Optional[str]) -> str:
    if tok is None or not tok[0].isalpha():
        return "END"
    t = tok.lower()
    if t in DET and t not in ("her", "that", "which", "what", "one"):
        return "DET"
    if t in PRON_OBJ:
        return "OBJ"
    if t in PREP or t in TO:
        return "PREP"
    if t in CONJ:
        return "CONJ"
    if t in BE or t in MODAL or t in PERF:
        return "AUX"
    return "OPEN"


def frame(text: str, char_start: int) -> Tuple[str, str]:
    """(left, right) frame classes of the word starting at `char_start` in `text`."""
    toks = [(m.group(), m.start()) for m in _TOKEN.finditer(text)]
    idx = next((i for i, (_, s) in enumerate(toks) if s >= char_start), None)
    if idx is None:
        return "START", "END"
    j = idx - 1
    while j >= 0 and toks[j][0].lower() in SKIP_LEFT:
        j -= 1
    left = _cls(toks[j][0]) if j >= 0 else "START"
    if left == "OPEN":
        # a determiner one or two modifiers back makes this a noun phrase ("a deep wound")
        for k in (j - 1, j - 2):
            if k < 0:
                break
            c = _cls(toks[k][0])
            if c == "DET":
                left = "DET_MOD"
                break
            if c != "OPEN":
                break
    right = _right_cls(toks[idx + 1][0]) if idx + 1 < len(toks) else "END"
    return left, right


def features(text: str, char_start: int) -> List[Tuple[str, ...]]:
    left, right = frame(text, char_start)
    # The left-only level treats "the ___" and "a deep ___" as one noun-phrase frame
    return [("LR", left, right), ("L", "DET" if left == "DET_MOD" else left), ("R", right)]


def coarse_role(text: str, char_start: int) -> Optional[str]:
    left, right = frame(text, char_start)
    if left in NOUN_LIKE:
        return "noun"
    if left in VERB_LIKE:
        return "verb"
    if left == "START" and right in ("DET", "OBJ"):
        return "verb"   # imperative: "Read the label", "Lead the way"
    return None


class SenseExample:
    """One corrected occurrence: the sense it belongs to and where the word sits in the sentence."""

    __slots__ = ("sense", "text", "char_start")

    def __init__(self, sense: int, text: str, char_start: int):
        self.sense, self.text, self.char_start = sense, text, char_start


def decide(
    examples: Sequence[SenseExample],
    senses: Sequence[int],
    text: str,
    char_start: int,
) -> Tuple[str, Optional[int], Dict]:
    """('choose', sense, info) | ('abstain', None, info) | ('undecided', None, info)."""
    occ = features(text, char_start)
    info: Dict = {"frame": occ[0][1:], "level": None, "votes": None}
    ex_feats = [(e.sense, features(e.text, e.char_start)) for e in examples]
    single = len(set(senses)) == 1
    for level, f in enumerate(occ):
        if single and f[0] == "R":
            break  # with nothing to contrast against, the right word alone is too weak to apply a correction
        votes: Dict[int, int] = {}
        for sense, feats in ex_feats:
            if feats[level] == f:
                votes[sense] = votes.get(sense, 0) + 1
        if not votes:
            continue
        ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            info.update({"level": f[0], "votes": votes})
            return "choose", ranked[0][0], info
    if single and examples:
        info["role"] = coarse_role(text, char_start)
        return "abstain", None, info
    return "undecided", None, info
