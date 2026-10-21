"""
Word-level multi-scale context profiles for Context-Robust Hopfield Retrieval.

Pure tensor functions (no model, no word lists):
    1. align backbone tokens to characters of the original text (follows the backbone tokenizer,
       e.g. F5-TTS's character/pinyin preprocessing which may insert or drop characters),
    2. pool token embeddings into word vectors,
    3. build left/right context vectors at several word radii plus a sentence vector,
    4. compare profiles,
    5. redistribute a group's baseline attention mass among its members (other groups unchanged).
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", flags=re.UNICODE)

Profile = Dict[str, torch.Tensor]


def word_char_spans(text: str) -> List[Tuple[int, int]]:
    """Character spans of the words of `text` (generic Unicode letter/digit runs, apostrophes kept)."""
    return [m.span() for m in _WORD_RE.finditer(text or "")]


def align_tokens_to_chars(text: str, token_chars: Sequence[str]) -> List[int]:
    """Map each backbone token to the index of the text character it encodes (-1 = inserted token).

    Monotonic greedy alignment: a token matches the next text character (case-insensitive); text
    characters the preprocessing dropped/rewrote are skipped with a short look-ahead; tokens with
    no counterpart (e.g. spaces inserted between segments) get -1.
    """
    out: List[int] = []
    i, n = 0, len(text)
    for tok in token_chars:
        t = (tok or "")[:1].lower()
        if not t:
            out.append(-1)
            continue
        if i < n and text[i].lower() == t:
            out.append(i)
            i += 1
            continue
        found = -1
        for j in range(i + 1, min(n, i + 4)):
            if text[j].lower() == t:
                found = j
                break
        if found >= 0:
            out.append(found)
            i = found + 1
        else:
            out.append(-1)
            if i < n and not text[i].isalnum() and not t.isalnum():
                i += 1  # punctuation rewritten by the preprocessing (e.g. ';' -> ',')
    return out


def proportional_token_chars(n_chars: int, n_tokens: int) -> List[int]:
    """Fallback token→char map when no tokenizer is available."""
    if n_tokens <= 0:
        return []
    if n_chars == n_tokens:
        return list(range(n_tokens))
    return [min(n_chars - 1, int((k + 0.5) * n_chars / n_tokens)) for k in range(n_tokens)]


def token_word_ids(token_chars: Sequence[int], spans: Sequence[Tuple[int, int]]) -> List[int]:
    """Word index of every token (-1 for spaces/punctuation/inserted tokens)."""
    char_to_word: Dict[int, int] = {}
    for w, (s, e) in enumerate(spans):
        for c in range(s, e):
            char_to_word[c] = w
    return [char_to_word.get(c, -1) if c >= 0 else -1 for c in token_chars]


def word_vectors(emb: torch.Tensor, tok_words: Sequence[int], n_words: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean token embedding per word → ([n_words, d], valid mask [n_words])."""
    if emb.dim() == 3:
        emb = emb[0]
    d = emb.shape[-1]
    vecs = torch.zeros(n_words, d, dtype=torch.float32)
    counts = torch.zeros(n_words, dtype=torch.float32)
    e = emb.detach().cpu().float()
    for t, w in enumerate(tok_words[: e.shape[0]]):
        if 0 <= w < n_words:
            vecs[w] += e[t]
            counts[w] += 1
    valid = counts > 0
    vecs[valid] = vecs[valid] / counts[valid].unsqueeze(-1)
    return vecs, valid


def build_profile(
    vecs: torch.Tensor,
    valid: torch.Tensor,
    target: int,
    radii: Sequence[int] = (1, 2, 4),
) -> Profile:
    """Left/right context vectors at each radius (1/k distance decay) + sentence vector.

    All words are kept (no stopword removal); the target word itself is excluded.
    Keys: "L{r}", "R{r}", "S". Sides without words are omitted.
    """
    n = vecs.shape[0]
    prof: Profile = {}
    for r in sorted(set(int(x) for x in radii if int(x) > 0)):
        for side, rng in (("L", range(target - 1, max(-1, target - r - 1), -1)),
                          ("R", range(target + 1, min(n, target + r + 1)))):
            acc, wsum = None, 0.0
            for j in rng:
                if not bool(valid[j]):
                    continue
                w = 1.0 / abs(j - target)
                acc = vecs[j] * w if acc is None else acc + vecs[j] * w
                wsum += w
            if acc is not None:
                prof[f"{side}{r}"] = F.normalize(acc / wsum, p=2, dim=-1)
    others = [j for j in range(n) if j != target and bool(valid[j])]
    if others:
        prof["S"] = F.normalize(vecs[others].mean(dim=0), p=2, dim=-1)
    return prof


def profile_from_embeddings(
    emb: torch.Tensor,
    text: str,
    char_span: Tuple[int, int],
    token_chars: Optional[Sequence[int]] = None,
    radii: Sequence[int] = (1, 2, 4),
) -> Optional[Profile]:
    """Context profile of the word occurrence at `char_span` in `text`."""
    spans = word_char_spans(text)
    if not spans:
        return None
    n_tok = emb.shape[1] if emb.dim() == 3 else emb.shape[0]
    if token_chars is None or len(token_chars) != n_tok:
        token_chars = proportional_token_chars(len(text), n_tok)
    cs, ce = int(char_span[0]), int(char_span[1])
    target = next((w for w, (s, e) in enumerate(spans) if s < ce and e > cs), None)
    if target is None:
        return None
    vecs, valid = word_vectors(emb, token_word_ids(token_chars, spans), len(spans))
    return build_profile(vecs, valid, target, radii)


def profile_similarity(p: Optional[Profile], q: Optional[Profile], weights: Optional[Dict[str, float]] = None) -> float:
    """Σ_r λ_r cos(C_r^p, C_r^q) over shared scales; λ uniform by default (normalised to sum 1)."""
    if not p or not q:
        return 0.0
    keys = [k for k in p if k in q and p[k].shape == q[k].shape]
    if not keys:
        return 0.0
    lam = {k: (weights or {}).get(k, 1.0) for k in keys}
    z = sum(lam.values()) or 1.0
    return float(sum(lam[k] * torch.dot(p[k].float(), q[k].float()) for k in keys) / z)


def ema_profile(old: Optional[Profile], new: Optional[Profile], decay: float) -> Optional[Profile]:
    """EMA merge of two profiles (scales present in only one side are taken as-is)."""
    if not old:
        return new
    if not new:
        return old
    out: Profile = {}
    for k in set(old) | set(new):
        if k in old and k in new and old[k].shape == new[k].shape:
            out[k] = F.normalize(old[k].float() * decay + new[k].float() * (1.0 - decay), p=2, dim=-1)
        else:
            out[k] = (new.get(k) if k in new else old[k]).float()
    return out


def group_preserving_attention(
    base_scores: torch.Tensor,
    group: Sequence[int],
    new_group_scores: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Redistribute the baseline attention mass of `group` using new (already β-scaled) scores.

        a_base = softmax(base_scores)
        A_G    = Σ_{j∈G} a_base_j
        a_new_i = A_G · softmax_G(new_group_scores)_i   for i ∈ G;   a_new_k = a_base_k otherwise

    Returns (a_base, a_new) over all M entries.
    """
    a_base = torch.softmax(base_scores.float(), dim=-1)
    a_new = a_base.clone()
    if len(group) == 0:
        return a_base, a_new
    idx = torch.tensor(list(group), dtype=torch.long)
    mass = a_base[idx].sum()
    a_new[idx] = mass * torch.softmax(new_group_scores.float(), dim=-1)
    return a_base, a_new
