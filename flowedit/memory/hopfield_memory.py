"""
Modern Hopfield Memory — Stage 3 of FlowEdit (arXiv:2606.20518).

Paper Reference: Section 3.2 & Equations 5, 6, 7.
Continuous associative memory storing learned pronunciation corrections.

Mathematical Formulation:
    Key Storage (Eq. 5):
        K_i = pool(c_I) ∈ R^d
        V_i = pool(δ*_I) ∈ R^d
    Retrieval (Eq. 6):
        Mem(Q) = softmax(β Q K^T) V, where β = 1/√d
    Similarity Gating (Eq. 7):
        c_hat = c + σ(max_j(β Q K_j^T) - τ) ⊙ Mem(Q)

Enhanced with:
    1. EMA sense clusters: every entry is one pronunciation sense of a word. Repeated corrections
       of the same sense are merged by an exponential moving average of the backbone key K, the
       pooled value V, the word perturbation δ*_word and a contextual sense prototype P.
    2. Per-occurrence homograph disambiguation: each occurrence of a word in the input is encoded
       in its own clause-bounded context window and assigned to the nearest sense prototype, so
       the same spelling can resolve to different senses within one sentence.
    3. Backbone embedding spaces: keys and perturbations are stored per backbone space (XTTS
       d=1024 BPE, F5-TTS d=512 characters). Entries learned on another backbone are re-keyed
       with the local text encoder and never inject foreign perturbations; their sense prototype
       and pronunciation transfer unchanged because the sense space is backbone-independent.
    4. Multi-word simultaneous retrieval and exact BPE token-span localization.
"""

import os
import re
import math
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowedit.config import MemoryConfig

logger = logging.getLogger(__name__)

GRAMMAR_STOPWORDS = {
    'a', 'an', 'the',
    'is', 'are', 'am', 'was', 'were', 'be', 'been', 'being',
    'has', 'have', 'had', 'do', 'does', 'did',
    'of', 'in', 'on', 'at', 'to', 'for', 'with', 'from',
    'that', 'this', 'these', 'those', 'it', 'its',
    'and', 'or', 'but', 'nor', 'so', 'yet'
}

DIGIT_WORDS = {
    '0': ' zero ', '1': ' one ', '2': ' two ', '3': ' three ', '4': ' four ',
    '5': ' five ', '6': ' six ', '7': ' seven ', '8': ' eight ', '9': ' nine ',
}

_AUTO = "auto"


def broad_acoustic_hash(s: str) -> str:
    """Compute coarse acoustic/phonetic hash invariant to common ASR plosive/fricative confusions.

    Acoustically groups phonetically similar consonants and vowels:
    - Stops / Plosives / Labiodentals: b, p, d, t, v, f, w -> 1
    - Velar stops: g, k, c, q -> 2
    - Sibilants / Affricates: s, z, j, x -> 3
    - Liquids: l, r -> 4
    - Nasals: m, n -> 5
    - Vowels / Glides: a, e, i, o, u, y -> 0
    - Digraph reductions: 'rh' -> 'r', 'ph' -> 'f', 'th' -> 't', 'ck' -> 'k', 'sh' -> 's'
    - Aspirated stop reduction: bh -> b, dh -> d, kh -> k, gh -> g
    """
    s = re.sub(r'\d', lambda m: DIGIT_WORDS.get(m.group(0), ' '), s)
    s = re.sub(r'[^a-z]', '', s.lower())
    if not s:
        return ''
    s = re.sub(r'^rh', 'r', s)
    s = re.sub(r'ph', 'f', s)
    s = re.sub(r'th', 't', s)
    s = re.sub(r'ck', 'k', s)
    s = re.sub(r'sh', 's', s)
    s = re.sub(r'ch', 's', s)
    s = re.sub(r'zh', 'z', s)
    s = re.sub(r'([bdfgkt])h', r'\1', s)
    mapping = {
        'a': '0', 'e': '0', 'i': '0', 'o': '0', 'u': '0', 'y': '0',
        'b': '1', 'p': '1', 'd': '1', 't': '1', 'v': '1', 'f': '1', 'w': '1',
        'g': '2', 'k': '2', 'c': '2', 'q': '2',
        's': '3', 'z': '3', 'j': '3', 'x': '3',
        'l': '4', 'r': '4',
        'm': '5', 'n': '5',
    }
    out = []
    prev = None
    for ch in s:
        code = mapping.get(ch, ch)
        if code != prev:
            out.append(code)
            prev = code
    return ''.join(out)


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Calculate normalized Levenshtein similarity ratio between two strings (0.0 to 1.0)."""
    c1 = re.sub(r'[^a-z0-9]', '', s1.lower())
    c2 = re.sub(r'[^a-z0-9]', '', s2.lower())
    if not c1 or not c2:
        return 1.0 if c1 == c2 else 0.0
    if c1 == c2:
        return 1.0
    len1, len2 = len(c1), len(c2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if c1[i - 1] == c2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    dist = dp[len1][len2]
    return 1.0 - (dist / max(len1, len2))


def _resample_last_dim(t: Optional[torch.Tensor], target_dim: int, normalize_1d: bool = False) -> Optional[torch.Tensor]:
    """Linearly resample the channel dimension (used only for legacy entries without a local re-key)."""
    if t is None or t.shape[-1] == target_dim:
        return t
    shape = list(t.shape)
    flat = t.reshape(-1, 1, shape[-1]).float()
    res = F.interpolate(flat, size=target_dim, mode='linear', align_corners=True)
    out = res.reshape(shape[:-1] + [target_dim]).to(dtype=t.dtype, device=t.device)
    if normalize_1d and out.dim() == 1:
        out = F.normalize(out, p=2, dim=-1)
    return out


@dataclass
class MemoryEntry:
    """A single pronunciation sense cluster in Modern Hopfield Memory.

    `key`, `value`, `word_delta` and `full_delta` are the view in the owning memory's backbone
    space. `space_data` keeps the per-space tensors of every backbone that has learned this sense,
    and `delta_space` names the space the local perturbations came from (None = no usable δ*).
    """
    key: torch.Tensor           # K_i ∈ R^d (Gaussian context pooled base embedding)
    value: torch.Tensor         # V_i ∈ R^d (pooled latent perturbation δ*)
    word: str                   # Text representation of target word
    carrier_text: str           # Context sentence
    token_indices: List[int]    # Token indices I
    language: str = "en"
    access_count: int = 1
    last_access_step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    full_delta: Optional[torch.Tensor] = None   # Full δ* ∈ R^[1, S, d]
    word_delta: Optional[torch.Tensor] = None   # Canonical word-level δ* ∈ R^[1, L_word, d]
    carrier_texts: List[str] = field(default_factory=list)  # Context sentences merged into this cluster
    phonetic_text: Optional[str] = None                     # Acoustic pronunciation representation from reference audio
    sense_prototype: Optional[torch.Tensor] = None          # EMA of contextual occurrence embeddings (backbone-independent)
    space_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    delta_space: Optional[str] = None


@dataclass
class MatchedWordSpan:
    """A single word span matched by Hopfield Memory during inference."""
    word: str
    start_pos: int
    end_pos: int
    entry_index: int
    score: float
    gate_value: float
    carrier_text: str
    phonetic_text: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    has_delta: bool = False
    sense_similarity: Optional[float] = None


@dataclass
class RetrievalResult:
    """Structured result from Modern Hopfield Memory retrieval."""
    retrieved_delta: torch.Tensor   # Mem(Q) ∈ R^[1, S, d] assembled with all matched word perturbations
    gate_values: torch.Tensor       # σ(max_j(β Q K_j^T) - τ) ∈ R^[1, S, 1]
    similarities: torch.Tensor      # Cosine similarity scores
    top_matches: List[Tuple[str, float]]
    is_active: bool
    matched_spans: List[MatchedWordSpan] = field(default_factory=list)
    matched_full_delta: Optional[torch.Tensor] = None  # Full δ* from top entry [1, S_orig, d]
    matched_word: Optional[str] = None                 # Word from top matching entry
    matched_token_indices: Optional[List[int]] = None   # Token indices from top entry
    matched_carrier_text: Optional[str] = None          # Carrier text from top entry


def _norm_pron(word: str, phonetic: Optional[str]) -> Optional[str]:
    """Normalised pronunciation label, or None when it carries no information beyond the spelling."""
    if not phonetic:
        return None
    p = re.sub(r"\s+", " ", phonetic.strip().lower())
    if not p or p == (word or "").strip().lower():
        return None
    return p


class HopfieldMemory(nn.Module):
    """Modern Continuous Hopfield Associative Memory for Lifelong Pronunciation Adaptation."""

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        embedding_dim: int = 512,
        embedding_space: Optional[str] = None,
        text_encoder: Optional[Callable[[str, str], torch.Tensor]] = None,
        sense_encoder: Any = _AUTO,
        sense_lexicon: Any = _AUTO,
    ):
        super().__init__()
        self.config = config or MemoryConfig()
        self.d = embedding_dim
        self.space = embedding_space or f"d{embedding_dim}"
        self.beta = self.config.hopfield_beta or (1.0 / math.sqrt(self.d))
        self.max_entries = self.config.max_entries
        self.homograph_threshold = self.config.homograph_sim_threshold
        self.dedup_ema = self.config.dedup_ema_decay
        self.context_window = self.config.context_window
        self.context_char_radius = self.config.context_char_radius
        self.context_sigma = self.config.context_sigma

        # Contextual averaging weights for memory key computation
        # K_i = α · word_emb + β · local_context + γ · global_context
        self.context_key_word_weight = self.config.context_key_word_weight
        self.context_key_local_weight = self.config.context_key_local_weight
        self.context_key_global_weight = self.config.context_key_global_weight

        # Gate threshold τ (Paper Section 3.2: calibrated to 4.0 for robust target word matching across sentences)
        self.gate_threshold = nn.Parameter(
            torch.tensor(float(self.config.gate_threshold_init), dtype=torch.float32)
        )

        self.entries: List[MemoryEntry] = []
        self.current_step = 0
        self._text_encoder = text_encoder
        self._sense_encoder_spec = sense_encoder
        self._sense_lexicon_spec = sense_lexicon
        self._rekey_cache: Dict[Tuple[str, str, Tuple[int, int]], torch.Tensor] = {}

    @property
    def num_entries(self) -> int:
        return len(self.entries)

    def is_empty(self) -> bool:
        return len(self.entries) == 0

    # ─────────────────────────────────────────────────────────────────────────
    # Encoders
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def sense_encoder(self):
        """Backbone-independent contextual encoder, or None when disabled/unavailable."""
        spec = self._sense_encoder_spec
        if spec is _AUTO:
            from flowedit.memory.sense_encoder import get_sense_encoder
            spec = get_sense_encoder(self.config)
            self._sense_encoder_spec = spec
        if not spec:
            return None
        if hasattr(spec, "available") and not spec.available:
            return None
        return spec

    @property
    def sense_lexicon(self):
        """Pretrained homograph sense lexicon (shared, adapts to corrections), or None."""
        spec = self._sense_lexicon_spec
        if spec is _AUTO:
            from flowedit.memory.sense_lexicon import get_sense_lexicon
            spec = get_sense_lexicon(self.config)
            self._sense_lexicon_spec = spec
        return spec or None

    def _lexicon_word(self, word: str) -> bool:
        lex = self.sense_lexicon
        return lex is not None and len(lex.labels(word)) > 1

    def _entry_label(self, entry: MemoryEntry) -> Optional[str]:
        """Lexicon sense an entry belongs to (from its spelling, else from its carrier context)."""
        if not self._lexicon_word(entry.word):
            return None
        label = entry.metadata.get("lexicon_label")
        lex = self.sense_lexicon
        if label in lex.labels(entry.word):
            return label
        label = lex.label_for_spelling(entry.word, entry.phonetic_text)
        if label is None and not self._is_unconditional(entry.word, entry.carrier_text):
            span = self._pick_span(entry.carrier_text, entry.word, char_span=entry.metadata.get("char_span"))
            if span is not None:
                label = lex.predict(entry.carrier_text, [(entry.word.lower(), span[0], span[1])])[0][0]
        if label is not None:
            entry.metadata["lexicon_label"] = label
        return label

    def attach_text_encoder(self, text_encoder: Callable[[str, str], torch.Tensor]) -> None:
        """Attach the owning backbone's text encoder and re-key entries learned on other backbones."""
        self._text_encoder = text_encoder
        for entry in self.entries:
            if entry.space_data.get(self.space) is None:
                self._localize(entry)

    def _occurrence_vectors(self, text: str, spans: List[Tuple[int, int]]) -> Optional[torch.Tensor]:
        enc = self.sense_encoder
        if enc is None or not text or not spans:
            return None
        try:
            return enc.encode(text, spans)
        except Exception as e:
            logger.warning(f"Sense encoding failed for '{text[:60]}': {e}")
            return None

    @staticmethod
    def _is_unconditional(word: str, carrier_text: Optional[str]) -> bool:
        """An entry learned without sentence context applies to every occurrence of its word."""
        c = (carrier_text or "").strip().strip(".!?,;:").strip().lower()
        return not c or c == (word or "").strip().lower()

    def _pick_span(
        self,
        text: str,
        word: str,
        char_span: Optional[Tuple[int, int]] = None,
        token_indices: Optional[List[int]] = None,
        total_tokens: Optional[int] = None,
    ) -> Optional[Tuple[int, int]]:
        """Character span of the corrected occurrence of `word` in `text`."""
        spans = self._find_word_char_spans(text, word)
        if not spans:
            return None
        if char_span is not None:
            return min(spans, key=lambda s: abs(s[0] - int(char_span[0])))
        if token_indices and len(spans) > 1:
            n_tok = total_tokens or len(text)
            est = min(token_indices) * len(text) / max(1, n_tok)
            return min(spans, key=lambda s: abs(s[0] - est))
        return spans[0]

    def _entry_prototype(self, entry: MemoryEntry) -> Optional[torch.Tensor]:
        """Sense prototype of an entry; legacy entries derive it from their carrier sentences (EMA)."""
        if entry.sense_prototype is not None:
            return entry.sense_prototype
        if self._is_unconditional(entry.word, entry.carrier_text):
            return None
        carriers = [c for c in (entry.carrier_texts or [entry.carrier_text]) if c]
        proto = None
        for i, c in enumerate(carriers):
            span = self._pick_span(
                c, entry.word,
                char_span=entry.metadata.get("char_span") if i == 0 else None,
                token_indices=entry.token_indices if i == 0 else None,
            )
            if span is None:
                continue
            vec = self._occurrence_vectors(c, [span])
            if vec is None:
                return None
            v = vec[0]
            proto = v if proto is None else F.normalize(self.dedup_ema * proto + (1.0 - self.dedup_ema) * v, p=2, dim=-1)
        entry.sense_prototype = proto
        return proto

    # ─────────────────────────────────────────────────────────────────────────
    # Backbone context keys
    # ─────────────────────────────────────────────────────────────────────────

    def compute_context_key(
        self,
        embeddings: torch.Tensor,
        token_indices: List[int],
        carrier_text: Optional[str] = None,
    ) -> torch.Tensor:
        """Compute word-centric context key K_i ∈ R^d using three-component contextual averaging.

        Blends three embedding components for sense-level generalization and homograph disambiguation:
            K_i = α · word_emb + β · local_context + γ · global_context

        Where:
            - word_emb: mean of target word token embeddings (primary signal)
            - local_context: Gaussian-weighted mean of ±R neighboring tokens, excluding
              the target tokens themselves, capturing disambiguating local context
              (e.g., "river" near "bank" vs "account" near "bank")
            - global_context: mean of all sequence tokens (coarse sentence signal)

        The configurable weights (α, β, γ) default to (0.70, 0.20, 0.10), keeping the
        word embedding dominant while incorporating enough context for sense discrimination.
        """
        if embeddings.dim() == 3:
            emb = embeddings[0]  # [S, d]
        else:
            emb = embeddings     # [S, d]

        seq_len, dim = emb.shape

        if not token_indices:
            return F.normalize(emb.mean(dim=0), p=2, dim=-1)

        # Valid target token span
        min_idx = max(0, min(token_indices))
        max_idx = min(seq_len - 1, max(token_indices))

        # 1. Target word embedding: mean of target token embeddings
        word_emb = emb[min_idx:max_idx + 1].mean(dim=0)

        # 2. Local context: Gaussian-weighted average of neighboring tokens (excluding target span)
        radius = max(1, int(self.context_char_radius))
        sigma = max(1.0, float(self.context_sigma))
        local_start = max(0, min_idx - radius)
        local_end = min(seq_len, max_idx + 1 + radius)

        # Collect neighbor token embeddings and their Gaussian weights
        local_positions = []
        for pos in range(local_start, local_end):
            if pos < min_idx or pos > max_idx:  # Exclude target tokens
                local_positions.append(pos)

        if local_positions:
            # Compute Gaussian weights centered on the target span midpoint
            center = (min_idx + max_idx) / 2.0
            neighbor_embs = emb[local_positions]  # [N_local, d]
            distances = torch.tensor(
                [abs(pos - center) for pos in local_positions],
                device=emb.device, dtype=emb.dtype,
            )
            gauss_weights = torch.exp(-(distances ** 2) / (2.0 * sigma ** 2))
            gauss_weights = gauss_weights / gauss_weights.sum()
            local_context = (gauss_weights.unsqueeze(-1) * neighbor_embs).sum(dim=0)  # [d]
        else:
            # Fallback: if no neighbors exist (very short sequence), use the word embedding itself
            local_context = word_emb

        # 3. Global context: mean of all sequence tokens
        global_context = emb.mean(dim=0)

        # Three-component contextual averaging blend
        alpha = self.context_key_word_weight
        beta = self.context_key_local_weight
        gamma = self.context_key_global_weight
        sense_key = alpha * word_emb + beta * local_context + gamma * global_context
        return F.normalize(sense_key, p=2, dim=-1)

    def _rekey(self, entry: MemoryEntry) -> Optional[torch.Tensor]:
        """Recompute an entry's key in this backbone's space from its carrier sentence."""
        if self._text_encoder is None:
            return None
        carrier = entry.carrier_text or entry.word
        span = self._pick_span(carrier, entry.word, char_span=entry.metadata.get("char_span"))
        cache_key = (carrier, entry.word.strip().lower(), span or (0, 0))
        cached = self._rekey_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            with torch.no_grad():
                emb = self._text_encoder(carrier, entry.language or "en")
            n_tok = emb.shape[1] if emb.dim() == 3 else emb.shape[0]
            tok = []
            if span is not None:
                ts, te = self._char_span_to_token_span(span[0], span[1], len(carrier), n_tok)
                tok = list(range(ts, te))
            key = self.compute_context_key(emb, tok).detach().cpu().float()
        except Exception as e:
            logger.warning(f"Could not re-key '{entry.word}' in space {self.space}: {e}")
            return None
        if key.shape[-1] != self.d:
            return None
        self._rekey_cache[cache_key] = key
        return key

    def _localize(self, entry: MemoryEntry) -> None:
        """Populate the entry's local view (key/value/δ*) for this memory's backbone space."""
        local = entry.space_data.get(self.space)
        if local is not None:
            entry.key = F.normalize(local["key"].float(), p=2, dim=-1)
            entry.value = local["value"].float() if local.get("value") is not None else torch.zeros(self.d)
            entry.word_delta = local.get("word_delta")
            entry.full_delta = local.get("full_delta")
            entry.token_indices = list(local.get("token_indices") or entry.token_indices or [])
            entry.delta_space = self.space if entry.word_delta is not None else None
            return

        # Learned on another backbone: its perturbations are meaningless here and are never injected.
        entry.word_delta = None
        entry.full_delta = None
        entry.value = torch.zeros(self.d)
        entry.delta_space = None
        key = self._rekey(entry)
        if key is None:
            foreign = next(iter(entry.space_data.values()), None)
            src = foreign["key"] if foreign is not None else entry.key
            key = _resample_last_dim(src.float(), self.d, normalize_1d=True)
        entry.key = F.normalize(key, p=2, dim=-1)

    def _store_local(self, entry: MemoryEntry) -> None:
        """Record the entry's local view as this backbone space's data."""
        entry.space_data[self.space] = {
            "key": entry.key.detach().cpu().float(),
            "value": entry.value.detach().cpu().float() if entry.value is not None else None,
            "word_delta": entry.word_delta if entry.delta_space == self.space else None,
            "full_delta": entry.full_delta if entry.delta_space == self.space else None,
            "token_indices": list(entry.token_indices or []),
        }

    def _extract_word_delta(
        self,
        full_delta: Optional[torch.Tensor],
        word: str,
        carrier_text: str = "",
        token_indices: Optional[List[int]] = None,
    ) -> Optional[torch.Tensor]:
        """Extract canonical per-character perturbation tensor for target word [1, L_word, d]."""
        if full_delta is None:
            return None

        fd = full_delta.detach().cpu().float()
        if fd.dim() == 2:
            fd = fd.unsqueeze(0)

        S = fd.shape[1]
        word_len = len(word)

        if token_indices and len(token_indices) > 0:
            idx_min = max(0, min(token_indices))
            idx_max = min(S, max(token_indices) + 1)
            extracted = fd[:, idx_min:idx_max, :].clone()
            if extracted.shape[1] > 0:
                return extracted

        if carrier_text:
            idx = carrier_text.lower().find(word.lower())
            if idx >= 0:
                end_idx = min(S, idx + word_len)
                extracted = fd[:, idx:end_idx, :].clone()
                if extracted.shape[1] > 0:
                    return extracted

        # Fallback: slice beginning of full delta up to word length
        return fd[:, :min(S, max(1, word_len)), :].clone()

    # ─────────────────────────────────────────────────────────────────────────
    # Write (EMA sense clustering)
    # ─────────────────────────────────────────────────────────────────────────

    def _find_merge_target(
        self,
        word: str,
        key_norm: torch.Tensor,
        proto: Optional[torch.Tensor],
        pron: Optional[str],
        unconditional: bool,
        label: Optional[str] = None,
    ) -> Tuple[int, float]:
        """Best same-sense cluster for a new correction, or (-1, best_similarity).

        Pronunciation decides first: two different learned pronunciations are different senses by
        definition and are never merged; equal pronunciations are the same sense. When either
        pronunciation is unknown, the contextual sense prototype (or, without an encoder, the
        backbone context key) must reach the merge threshold.
        """
        best_idx, best_sim, best_any = -1, -1.0, -1.0
        w_norm = word.strip().lower()
        for idx, entry in enumerate(self.entries):
            if entry.word.strip().lower() != w_norm:
                continue
            e_pron = _norm_pron(entry.word, entry.phonetic_text)
            e_proto = None if unconditional else self._entry_prototype(entry)
            if proto is not None and e_proto is not None and e_proto.shape == proto.shape:
                sim = float(torch.dot(proto, e_proto.to(proto)))
                threshold = self.config.sense_merge_threshold
            else:
                sim = float(torch.dot(key_norm, F.normalize(entry.key.cpu().float(), p=2, dim=-1)))
                threshold = self.homograph_threshold
            best_any = max(best_any, sim)

            e_label = self._entry_label(entry) if label is not None else None
            if label is not None and e_label is not None:
                compatible = label == e_label
            elif pron is not None and e_pron is not None:
                compatible = pron == e_pron
            elif unconditional and self._is_unconditional(entry.word, entry.carrier_text):
                compatible = True
            else:
                compatible = sim >= threshold
            if compatible and sim > best_sim:
                best_idx, best_sim = idx, sim
        return best_idx, (best_sim if best_idx >= 0 else best_any)

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        word: str,
        carrier_text: str = "",
        token_indices: Optional[List[int]] = None,
        language: str = "en",
        metadata: Optional[Dict[str, Any]] = None,
        full_delta: Optional[torch.Tensor] = None,
        word_delta: Optional[torch.Tensor] = None,
        phonetic_text: Optional[str] = None,
        char_span: Optional[Tuple[int, int]] = None,
        sense_vector: Optional[torch.Tensor] = None,
    ) -> Tuple[int, str]:
        """Store or EMA-merge a correction into its sense cluster (Paper Section 3.2).

        Sense-level EMA (decay ρ, warm-up decay while the cluster is young):
            K ← norm(ρ K + (1-ρ) K_new)      V ← ρ V + (1-ρ) V_new
            δ*_word ← ρ δ*_word + (1-ρ) δ*_new
            P ← norm(ρ P + (1-ρ) P_new)      (contextual sense prototype)
        A correction that matches no existing sense of the word becomes a new sense cluster
        (e.g. "lead" metal vs "lead" leader).
        """
        self.current_step += 1
        key_1d = key.squeeze().detach().cpu().float()
        val_1d = value.squeeze().detach().cpu().float()

        if key_1d.dim() > 1:
            key_1d = key_1d.mean(dim=0)
        if val_1d.dim() > 1:
            val_1d = val_1d.mean(dim=0)

        key_norm = F.normalize(key_1d, p=2, dim=-1)

        # Determine canonical word_delta
        if word_delta is not None:
            cur_word_delta = word_delta.detach().cpu().float()
            if cur_word_delta.dim() == 2:
                cur_word_delta = cur_word_delta.unsqueeze(0)
        else:
            cur_word_delta = self._extract_word_delta(full_delta, word, carrier_text, token_indices)

        stored_full_delta = full_delta.detach().cpu().float() if full_delta is not None else None
        meta = dict(metadata or {})
        unconditional = self._is_unconditional(word, carrier_text)

        occ_span = None
        if carrier_text and not unconditional:
            occ_span = self._pick_span(carrier_text, word, char_span=char_span, token_indices=token_indices)
            if occ_span is not None:
                meta["char_span"] = [int(occ_span[0]), int(occ_span[1])]

        proto = None
        if sense_vector is not None:
            proto = F.normalize(sense_vector.detach().cpu().float().reshape(-1), p=2, dim=-1)
        elif occ_span is not None:
            vec = self._occurrence_vectors(carrier_text, [occ_span])
            proto = vec[0] if vec is not None else None

        pron = _norm_pron(word, phonetic_text)

        # Homographs known to the sense lexicon: a spelling that denotes one of its senses is a
        # labelled example (the lexicon learns this context); any other spelling is a custom
        # pronunciation for the sense the lexicon assigns to this context.
        new_label = None
        if self._lexicon_word(word) and occ_span is not None:
            lex = self.sense_lexicon
            new_label = lex.label_for_spelling(word, phonetic_text)
            if new_label is not None:
                lex.learn(word, carrier_text, occ_span, new_label)
            else:
                new_label = lex.predict(carrier_text, [(word.lower(), occ_span[0], occ_span[1])])[0][0]
            meta["lexicon_label"] = new_label

        if self.entries:
            best_idx, best_sim = self._find_merge_target(word, key_norm, proto, pron, unconditional, new_label)
            if best_idx >= 0:
                matched = self.entries[best_idx]
                if matched.access_count <= self.config.ema_warmup_count:
                    decay = float(self.config.ema_warmup_decay)
                else:
                    decay = float(self.dedup_ema)
                alpha = 1.0 - decay
                has_local = self.space in matched.space_data
                same_space = matched.delta_space == self.space

                # 1. Key EMA average (normalized, local backbone space)
                matched.key = F.normalize(matched.key.cpu().float() * decay + key_norm * alpha, p=2, dim=-1)

                # 2. Value EMA average (only blend values learned in the same space)
                if has_local:
                    matched.value = matched.value.cpu().float() * decay + val_1d * alpha
                else:
                    matched.value = val_1d

                # 3. Canonical word_delta EMA average
                if cur_word_delta is not None:
                    if same_space and matched.word_delta is not None and matched.word_delta.shape == cur_word_delta.shape:
                        matched.word_delta = matched.word_delta * decay + cur_word_delta * alpha
                    else:
                        matched.word_delta = cur_word_delta
                    matched.delta_space = self.space
                if stored_full_delta is not None:
                    matched.full_delta = stored_full_delta

                # 4. Contextual sense prototype EMA (backbone-independent)
                if proto is not None:
                    old = None if unconditional else self._entry_prototype(matched)
                    if old is not None and old.shape == proto.shape:
                        matched.sense_prototype = F.normalize(old * decay + proto * alpha, p=2, dim=-1)
                    else:
                        matched.sense_prototype = proto

                matched.access_count += 1
                matched.last_access_step = self.current_step
                meta.pop("char_span", None)
                matched.metadata.update(meta)

                if pron is not None or _norm_pron(matched.word, matched.phonetic_text) is None:
                    if phonetic_text:
                        matched.phonetic_text = phonetic_text

                if carrier_text:
                    if matched.carrier_texts is None:
                        matched.carrier_texts = [matched.carrier_text] if matched.carrier_text else []
                    if carrier_text not in matched.carrier_texts:
                        matched.carrier_texts.append(carrier_text)

                self._store_local(matched)
                logger.info(
                    f"✓ Hopfield Memory EMA-merged '{word}' into sense cluster {best_idx} "
                    f"(N={matched.access_count}, sim={best_sim:.4f}, decay={decay:.2f}, "
                    f"phonetic='{matched.phonetic_text}')"
                )
                return best_idx, "averaged"
            if best_sim > -1.0:
                logger.info(
                    f"✓ Hopfield Memory: new sense cluster for '{word}' "
                    f"(best same-word sim={best_sim:.4f}, phonetic='{phonetic_text}')"
                )

        # LRU Pruning if capacity exceeded (Paper Section 3.2: M_max = 500)
        if len(self.entries) >= self.max_entries:
            lru_idx = min(range(len(self.entries)), key=lambda i: self.entries[i].last_access_step)
            evicted = self.entries.pop(lru_idx)
            logger.info(f"Hopfield Memory Capacity Reached ({self.max_entries}). Evicted LRU entry '{evicted.word}'")

        carrier_list = [carrier_text] if carrier_text else []
        entry = MemoryEntry(
            key=key_norm,
            value=val_1d,
            word=word,
            carrier_text=carrier_text,
            token_indices=token_indices or [],
            language=language,
            access_count=1,
            last_access_step=self.current_step,
            metadata=meta,
            full_delta=stored_full_delta,
            word_delta=cur_word_delta,
            carrier_texts=carrier_list,
            phonetic_text=phonetic_text,
            sense_prototype=proto,
            delta_space=self.space if cur_word_delta is not None else None,
        )
        self._store_local(entry)
        self.entries.append(entry)
        idx = len(self.entries) - 1
        logger.info(f"✓ Hopfield Memory Stored: '{word}' at index {idx} (phonetic='{phonetic_text}', Total Entries: {len(self.entries)})")
        return idx, "inserted"

    def add_entry(self, entry: MemoryEntry):
        """Add an entry from another memory instance, localising it to this backbone space."""
        space_data = {
            s: dict(d) for s, d in (getattr(entry, "space_data", None) or {}).items()
        }
        if not space_data:
            src_space = self.space if entry.key.shape[-1] == self.d else f"legacy:d{entry.key.shape[-1]}"
            space_data[src_space] = {
                "key": entry.key.clone().detach().cpu().float(),
                "value": entry.value.clone().detach().cpu().float() if entry.value is not None else None,
                "word_delta": entry.word_delta,
                "full_delta": entry.full_delta,
                "token_indices": list(entry.token_indices or []),
            }

        self.current_step += 1
        new_entry = MemoryEntry(
            key=entry.key.clone().detach().cpu().float(),
            value=torch.zeros(self.d),
            word=entry.word,
            carrier_text=entry.carrier_text,
            token_indices=list(entry.token_indices or []),
            language=entry.language,
            access_count=entry.access_count,
            last_access_step=self.current_step,
            metadata=dict(entry.metadata or {}),
            carrier_texts=list(getattr(entry, "carrier_texts", []) or ([entry.carrier_text] if entry.carrier_text else [])),
            phonetic_text=entry.phonetic_text,
            sense_prototype=getattr(entry, "sense_prototype", None),
            space_data=space_data,
        )
        self._localize(new_entry)
        self.entries.append(new_entry)
        logger.info(f"✓ Added entry for '{entry.word}' into Hopfield Memory (space={self.space}, total={len(self.entries)})")

    # ─────────────────────────────────────────────────────────────────────────
    # Retrieval (per-occurrence sense selection)
    # ─────────────────────────────────────────────────────────────────────────

    def _find_word_char_spans(self, text: str, word: str) -> List[Tuple[int, int]]:
        """Find all character-level start and end spans of a word using word boundaries."""
        text_clean = text.strip()
        word_clean = word.strip()
        if not text_clean or not word_clean:
            return []

        # Try word-boundary regex first to avoid false substring matches inside other words
        pattern = r'\b' + re.escape(word_clean) + r'\b'
        matches = [m.span() for m in re.finditer(pattern, text_clean, flags=re.IGNORECASE)]
        if matches:
            return matches

        # Fallback to substring matching if word boundary returns empty
        pattern = re.escape(word_clean)
        return [m.span() for m in re.finditer(pattern, text_clean, flags=re.IGNORECASE)]

    def _char_span_to_token_span(
        self,
        char_start: int,
        char_end: int,
        total_chars: int,
        total_tokens: int,
    ) -> Tuple[int, int]:
        """Map character span [char_start, char_end] to token span [tok_start, tok_end]."""
        if total_chars <= 0 or total_tokens <= 0:
            return (0, total_tokens)

        if total_chars == total_tokens:
            # 1:1 character-level tokenization (e.g. F5-TTS)
            return (max(0, char_start), min(total_tokens, char_end))

        # BPE tokenization: proportional mapping with token boundary rounding
        ratio = float(total_tokens) / float(total_chars)
        tok_start = int(math.floor(char_start * ratio))
        tok_end = int(math.ceil(char_end * ratio))
        tok_start = max(0, min(total_tokens - 1, tok_start))
        tok_end = max(tok_start + 1, min(total_tokens, tok_end))
        return (tok_start, tok_end)

    def _select_sense(
        self,
        candidates: List[Tuple[int, MemoryEntry]],
        occ_vec: Optional[torch.Tensor],
        occ_key_fn: Callable[[], torch.Tensor],
    ) -> Optional[Tuple[int, MemoryEntry, float]]:
        """Assign one occurrence to a sense cluster of its word.

        - Several senses with different pronunciations: the most similar prototype wins if it
          reaches `sense_min_similarity` (relative decision between learned senses).
        - A single learned sense: applies only if similarity reaches `sense_match_threshold`.
        - Entries learned without sentence context apply when no contextual sense claims the
          occurrence.
        Without the contextual encoder the backbone context keys are compared instead.
        """
        unconditional = [(i, e) for i, e in candidates if self._is_unconditional(e.word, e.carrier_text)]
        contextual = [(i, e) for i, e in candidates if not self._is_unconditional(e.word, e.carrier_text)]

        if contextual:
            protos = [self._entry_prototype(e) for _, e in contextual] if occ_vec is not None else []
            use_protos = occ_vec is not None and all(p is not None and p.shape == occ_vec.shape for p in protos)
            if use_protos:
                sims = [float(torch.dot(occ_vec, p.to(occ_vec))) for p in protos]
                single_thr = self.config.sense_match_threshold
                multi_thr = self.config.sense_min_similarity
            else:
                occ_key = occ_key_fn()
                sims = [
                    float(torch.dot(occ_key, F.normalize(e.key.to(occ_key), p=2, dim=-1)))
                    for _, e in contextual
                ]
                single_thr = multi_thr = self.homograph_threshold
            if sims:
                classes = {
                    _norm_pron(e.word, e.phonetic_text) or f"#{i}" for i, e in contextual
                }
                threshold = multi_thr if len(classes) > 1 else single_thr
                j = max(range(len(sims)), key=lambda k: sims[k])
                if sims[j] >= threshold:
                    m_idx, entry = contextual[j]
                    return m_idx, entry, sims[j]

        if unconditional:
            m_idx, entry = max(unconditional, key=lambda c: c[1].access_count)
            return m_idx, entry, 1.0
        return None

    def _span_pronunciation(self, entry: MemoryEntry, text: str, char_start: int) -> Optional[str]:
        """Learned pronunciation of a matched occurrence (memory first, then registered spellings)."""
        if _norm_pron(entry.word, entry.phonetic_text):
            return entry.phonetic_text.strip()
        try:
            from flowedit.memory.s3_storage import s3_spelling_store
            spelled = s3_spelling_store.resolve_correction(text, entry.word, char_offset=char_start)
        except Exception:
            spelled = None
        if _norm_pron(entry.word, spelled):
            return spelled.strip()
        return entry.phonetic_text

    def retrieve(
        self,
        query_embeddings: torch.Tensor,
        target_token_indices: Optional[List[int]] = None,
        text: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        language: str = "en",
    ) -> RetrievalResult:
        """Stage 3: Multi-Word Contextual Retrieval using Modern Hopfield Network (Paper Eq. 6 & 7).

        Computes context-aware similarities, applies similarity gating σ(max_j(β Q K_j^T) - τ),
        and assigns every occurrence of a stored word to its own sense cluster before injecting
        that cluster's correction at the occurrence's token span.
        """
        device = query_embeddings.device
        dtype = query_embeddings.dtype
        if query_embeddings.dim() == 3:
            Q = query_embeddings[0]  # [S, d]
        else:
            Q = query_embeddings     # [S, d]

        seq_len, dim = Q.shape

        valid_entries = self.entries
        if any(e.key.shape[-1] != dim for e in valid_entries):
            # Query dimension differs from this memory's space (only in ad-hoc tests / misuse)
            valid_entries = [
                MemoryEntry(**{**e.__dict__, "key": _resample_last_dim(e.key, dim, normalize_1d=True),
                               "value": _resample_last_dim(e.value, dim),
                               "word_delta": None, "full_delta": None, "delta_space": None})
                if e.key.shape[-1] != dim else e
                for e in valid_entries
            ]
        lexicon_occ: List[Tuple[str, int, int]] = []
        if text and self.sense_lexicon is not None:
            lexicon_occ = [o for o in self.sense_lexicon.occurrences(text) if self._lexicon_word(o[0])]
        if not valid_entries and not lexicon_occ:
            zeros_delta = torch.zeros_like(query_embeddings)
            zeros_gate = torch.zeros(1, seq_len, 1, device=device, dtype=dtype)
            return RetrievalResult(
                retrieved_delta=zeros_delta,
                gate_values=zeros_gate,
                similarities=torch.zeros(seq_len, 0, device=device, dtype=dtype),
                top_matches=[],
                is_active=False,
            )

        # 1. Stack memory keys K ∈ R^[M, d] and values V ∈ R^[M, d]
        if valid_entries:
            K = torch.stack([e.key.to(device=device, dtype=dtype) for e in valid_entries])   # [M, d]
            V = torch.stack([e.value.to(device=device, dtype=dtype) for e in valid_entries]) # [M, d]
        else:
            K = torch.zeros(1, dim, device=device, dtype=dtype)
            V = torch.zeros(1, dim, device=device, dtype=dtype)

        # 2. Compute contextual query vectors Q_ctx across sequence using Gaussian smoothing
        char_radius = max(1, min(int(self.context_char_radius), max(1, seq_len // 4)))
        char_sigma = max(1.0, float(self.context_sigma))
        kernel_size = 2 * char_radius + 1
        x_grid = torch.arange(-char_radius, char_radius + 1, device=device, dtype=dtype)
        gaussian_kernel = torch.exp(- (x_grid ** 2) / (2.0 * (char_sigma ** 2)))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        Q_unsq = Q.unsqueeze(0).transpose(1, 2)  # [1, d, S]
        Q_padded = F.pad(Q_unsq, (char_radius, char_radius), mode='replicate')
        kernel_weight = gaussian_kernel.view(1, 1, kernel_size).repeat(dim, 1, 1)
        Q_ctx = F.conv1d(Q_padded, kernel_weight, groups=dim).transpose(1, 2).squeeze(0)  # [S, d]

        # 3. Normalize Q_ctx and K for cosine dot-product
        Q_norm = F.normalize(Q_ctx, p=2, dim=-1)  # [S, d]
        K_norm = F.normalize(K, p=2, dim=-1)      # [M, d]

        # 4. Hopfield Attention & Similarity Scores: S_ij = β Q_i K_j^T (Paper Eq. 6)
        beta_scale = float(self.config.retrieval_score_scale) if self.beta < 1.0 else float(self.beta)
        scores = beta_scale * torch.matmul(Q_norm, K_norm.transpose(0, 1))  # [S, M]

        # 5. Similarity Gating (Paper Section 3.2 & Eq. 7)
        tau = self.gate_threshold.to(device=device, dtype=dtype)
        if not valid_entries:
            scores = torch.full_like(scores, -1e4)
        max_scores, max_indices = torch.max(scores, dim=-1)                   # [S]
        gate_seq = torch.sigmoid(max_scores - tau)                            # [S]
        floor = float(self.config.gate_activation_floor)
        gate_seq_clean = torch.where(gate_seq >= floor, gate_seq, torch.zeros_like(gate_seq))

        # 6. Per-occurrence sense selection & span injection in token space
        matched_spans: List[MatchedWordSpan] = []
        assembled_delta = torch.zeros(1, seq_len, dim, device=device, dtype=dtype)
        gate_map = torch.zeros(1, seq_len, 1, device=device, dtype=dtype)

        if text:
            total_chars = len(text)

            # Group entries by normalized target word (each group = the senses of one spelling)
            word_to_entries: Dict[str, List[Tuple[int, MemoryEntry]]] = {}
            for m_idx, entry in enumerate(valid_entries):
                word_to_entries.setdefault(entry.word.strip().lower(), []).append((m_idx, entry))

            occurrences: List[Tuple[str, int, int]] = []
            for w_norm, candidates in word_to_entries.items():
                if self._lexicon_word(w_norm):
                    continue
                for c_start, c_end in self._find_word_char_spans(text, candidates[0][1].word):
                    occurrences.append((w_norm, c_start, c_end))
            occurrences.sort(key=lambda o: o[1])

            # Exact BPE prefix boundaries, decoded once per text (not once per occurrence)
            prefix_ends: Optional[List[int]] = None
            if (occurrences or lexicon_occ) and tokenizer is not None and hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode") \
                    and total_chars != seq_len:
                try:
                    enc = tokenizer.encode(text, lang=language)
                    if isinstance(enc, list) and enc and isinstance(enc[0], list):
                        enc = enc[0]
                    prefix_ends = []
                    for i in range(1, len(enc) + 1):
                        try:
                            prefix_ends.append(len(tokenizer.decode(enc[:i])))
                        except Exception:
                            prefix_ends.append(prefix_ends[-1] if prefix_ends else 0)
                except Exception:
                    prefix_ends = None

            occ_vecs = self._occurrence_vectors(text, [(o[1], o[2]) for o in occurrences]) if occurrences else None
            processed_token_spans = set()

            def token_span(c_start: int, c_end: int) -> Tuple[int, int]:
                t_start = t_end = None
                if prefix_ends:
                    prev_len = 0
                    tok_indices = []
                    for idx, curr_len in enumerate(prefix_ends):
                        if curr_len > c_start and prev_len < c_end:
                            tok_indices.append(idx)
                        prev_len = curr_len
                    if tok_indices:
                        t_start = max(0, min(tok_indices))
                        t_end = min(seq_len, max(tok_indices) + 1)
                if t_start is None:
                    t_start, t_end = self._char_span_to_token_span(c_start, c_end, total_chars, seq_len)
                start = max(0, t_start)
                return start, min(seq_len, max(t_end, start + 1))

            def inject(entry: MemoryEntry, span_start: int, span_end: int) -> None:
                src_delta = entry.word_delta.to(device=device, dtype=dtype)
                src_delta = src_delta[0] if src_delta.dim() == 3 else src_delta
                assembled_delta[0, span_start:span_end, :] += self._align_delta(src_delta, span_end - span_start)
                gate_map[0, span_start:span_end, 0] = 1.0

            # 6a. Lexicon homographs: the pretrained (and correction-adapted) classifier picks the
            #     sense of every occurrence; learned entries of that sense supply their pronunciation/δ*
            if lexicon_occ:
                lex = self.sense_lexicon
                preds = lex.predict(text, lexicon_occ)
                for (w_norm, c_start, c_end), (label, prob) in zip(lexicon_occ, preds):
                    span_start, span_end = token_span(c_start, c_end)
                    same_sense = [
                        (i, e) for i, e in word_to_entries.get(w_norm, []) if self._entry_label(e) == label
                    ]
                    m_idx, entry = max(same_sense, key=lambda c: c[1].access_count) if same_sense else (-1, None)
                    pron = None
                    if entry is not None and _norm_pron(entry.word, entry.phonetic_text):
                        pron = entry.phonetic_text.strip()
                    elif prob >= self.config.sense_lexicon_min_confidence and self._respell_sense(w_norm, label):
                        pron = lex.respelling(w_norm, label)
                    has_delta = entry is not None and entry.delta_space == self.space and entry.word_delta is not None
                    logger.info(
                        f"[Sense Lexicon] '{text[c_start:c_end]}'@{c_start} → {label} (p={prob:.2f}) "
                        f"pronounce={pron or '(default reading)'}{' +δ*' if has_delta else ''}"
                    )
                    if not pron and not has_delta:
                        continue
                    processed_token_spans.add((span_start, span_end))
                    matched_spans.append(MatchedWordSpan(
                        word=text[c_start:c_end], start_pos=span_start, end_pos=span_end, entry_index=m_idx,
                        score=float(scores[max(0, span_start - 2):min(seq_len, span_end + 2), m_idx].max().item()) if m_idx >= 0 else 0.0,
                        gate_value=1.0, carrier_text=entry.carrier_text if entry is not None else "",
                        phonetic_text=pron, char_start=c_start, char_end=c_end,
                        has_delta=has_delta, sense_similarity=prob,
                    ))
                    if has_delta:
                        inject(entry, span_start, span_end)

            for o_idx, (w_norm, c_start, c_end) in enumerate(occurrences):
                span_start, span_end = token_span(c_start, c_end)
                if (span_start, span_end) in processed_token_spans:
                    continue

                occ_vec = occ_vecs[o_idx] if occ_vecs is not None else None
                chosen = self._select_sense(
                    word_to_entries[w_norm],
                    occ_vec,
                    lambda s=span_start, e=span_end: self.compute_context_key(Q, list(range(s, e))),
                )
                if chosen is None:
                    logger.info(f"[Hopfield Retrieve] '{text[c_start:c_end]}' @{c_start}: no learned sense matches this context")
                    continue
                m_idx, entry, sense_sim = chosen
                processed_token_spans.add((span_start, span_end))

                search_start = max(0, span_start - 2)
                search_end = min(seq_len, span_end + 2)
                s_occ = scores[search_start:search_end, m_idx].max().item()

                has_delta = entry.delta_space == self.space and entry.word_delta is not None
                matched_spans.append(
                    MatchedWordSpan(
                        word=entry.word,
                        start_pos=span_start,
                        end_pos=span_end,
                        entry_index=m_idx,
                        score=s_occ,
                        gate_value=1.0,
                        carrier_text=entry.carrier_text,
                        phonetic_text=self._span_pronunciation(entry, text, c_start),
                        char_start=c_start,
                        char_end=c_end,
                        has_delta=has_delta,
                        sense_similarity=sense_sim,
                    )
                )

                if has_delta:
                    inject(entry, span_start, span_end)

        # 7. Text-free continuous retrieval (pure embedding query, Paper Eq. 6-7)
        if not matched_spans:
            if not text and gate_seq_clean.max().item() > 0:
                attn_weights = torch.softmax(scores, dim=-1)  # [S, M]
                mem_q = torch.matmul(attn_weights, V)          # [S, d]
                assembled_delta = (gate_seq_clean.unsqueeze(-1) * mem_q).unsqueeze(0)  # [1, S, d]
                gate_map = gate_seq_clean.unsqueeze(0).unsqueeze(-1)  # [1, S, 1]
            else:
                assembled_delta = torch.zeros(1, seq_len, dim, device=device, dtype=dtype)
                gate_map = torch.zeros(1, seq_len, 1, device=device, dtype=dtype)

        is_active = (len(matched_spans) > 0) or (gate_map.max().item() > 0.10)

        # Diagnostic summary
        top_matches = [(span.word, span.score) for span in matched_spans]

        best_full_delta = None
        best_word = None
        best_token_indices = None
        best_carrier = None

        if matched_spans:
            with_entry = [sp for sp in matched_spans if sp.entry_index >= 0]
            if with_entry:
                best_entry = valid_entries[max(with_entry, key=lambda sp: sp.score).entry_index]
                best_full_delta = best_entry.full_delta
                best_word = best_entry.word
                best_token_indices = best_entry.token_indices
                best_carrier = best_entry.carrier_text
            logger.info(
                f"[Hopfield Retrieve] Active word corrections: "
                f"{[(s.word, s.char_start, f'sense={s.entry_index}', f'sim={s.sense_similarity:.3f}', f'phonetic={s.phonetic_text}', f'delta={s.has_delta}') for s in matched_spans]}"
            )
        elif valid_entries and gate_seq_clean.max().item() > 0:
            best_idx = max_indices[torch.argmax(max_scores).item()].item()
            best_entry = valid_entries[best_idx]
            top_matches.append((best_entry.word, max_scores.max().item()))
            best_full_delta = best_entry.full_delta
            best_word = best_entry.word
            best_token_indices = best_entry.token_indices
            best_carrier = best_entry.carrier_text

        return RetrievalResult(
            retrieved_delta=assembled_delta,
            gate_values=gate_map,
            similarities=scores,
            top_matches=top_matches,
            is_active=is_active,
            matched_spans=matched_spans,
            matched_full_delta=best_full_delta,
            matched_word=best_word,
            matched_token_indices=best_token_indices,
            matched_carrier_text=best_carrier,
        )

    def _respell_sense(self, word: str, label: str) -> bool:
        """The corpus-dominant sense is the word's default reading; only other senses are respelled."""
        counts = self.sense_lexicon.sense_counts(word)
        total = sum(counts.values())
        return not total or counts.get(label, 0) / total < self.config.sense_lexicon_default_share

    @staticmethod
    def _align_delta(src_delta: torch.Tensor, span_len: int) -> torch.Tensor:
        """Stretch/contract a [L, d] word perturbation onto `span_len` positions."""
        if src_delta.shape[0] == span_len:
            return src_delta
        if span_len == 1:
            return src_delta.mean(dim=0, keepdim=True)
        if src_delta.shape[0] == 1:
            return src_delta.repeat(span_len, 1)
        s_3d = src_delta.unsqueeze(0).transpose(1, 2)  # [1, d, L_src]
        return F.interpolate(s_3d.float(), size=span_len, mode='linear', align_corners=True).transpose(1, 2).squeeze(0).to(src_delta.dtype)

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        """Serialize memory entries (all backbone spaces + sense prototypes) to disk."""
        encoder_name = (self.config.sense_encoder_model or "").strip()
        entries = []
        for e in self.entries:
            if e.space_data.get(self.space) is not None or e.delta_space == self.space:
                self._store_local(e)
            entries.append({
                # Legacy top-level fields mirror this backbone's view for older readers
                "key": e.key.cpu(),
                "value": e.value.cpu(),
                "word": e.word,
                "carrier_text": e.carrier_text,
                "carrier_texts": e.carrier_texts or ([e.carrier_text] if e.carrier_text else []),
                "token_indices": e.token_indices,
                "language": e.language,
                "access_count": e.access_count,
                "last_access_step": e.last_access_step,
                "metadata": e.metadata,
                "full_delta": e.full_delta.cpu() if (e.full_delta is not None and e.delta_space == self.space) else None,
                "word_delta": e.word_delta.cpu() if (e.word_delta is not None and e.delta_space == self.space) else None,
                "phonetic_text": e.phonetic_text,
                "embedding_space": self.space,
                "space_data": e.space_data,
                "sense_prototype": e.sense_prototype.cpu() if e.sense_prototype is not None else None,
            })
        data = {
            "entries": entries,
            "current_step": self.current_step,
            "gate_threshold": self.gate_threshold.data.cpu(),
            "sense_encoder_model": encoder_name,
        }
        torch.save(data, filepath)
        logger.info(f"✓ Saved {len(self.entries)} Hopfield Memory entries to {filepath} (space={self.space})")

    def load(self, filepath: str) -> None:
        """Load serialized memory entries from disk with backward compatibility."""
        if not os.path.exists(filepath):
            return
        try:
            data = torch.load(filepath, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning(f"Could not load memory file {filepath}: {e}")
            return

        same_encoder = (data.get("sense_encoder_model") or "") == (self.config.sense_encoder_model or "").strip()
        loaded_entries = []
        for item in data.get("entries", []):
            k = item.get("key")
            if k is None or not isinstance(k, torch.Tensor):
                continue

            word = item.get("word", "")
            carrier_text = item.get("carrier_text", "")
            token_indices = item.get("token_indices", [])
            full_delta = item.get("full_delta", None)
            word_delta = item.get("word_delta", None)

            space_data = item.get("space_data") or {}
            if not space_data:
                # Legacy file: one backbone space, identified by its tag or by the key dimension
                src_space = item.get("embedding_space") or (self.space if k.shape[-1] == self.d else f"legacy:d{k.shape[-1]}")
                if word_delta is None and full_delta is not None and word:
                    word_delta = self._extract_word_delta(full_delta, word, carrier_text, token_indices)
                space_data = {src_space: {
                    "key": k.float(),
                    "value": item.get("value", k).float(),
                    "word_delta": word_delta,
                    "full_delta": full_delta,
                    "token_indices": token_indices,
                }}

            entry = MemoryEntry(
                key=k.float(),
                value=torch.zeros(self.d),
                word=word,
                carrier_text=carrier_text,
                token_indices=token_indices,
                language=item.get("language", "en"),
                access_count=item.get("access_count", 1),
                last_access_step=item.get("last_access_step", 0),
                metadata=item.get("metadata", {}) or {},
                carrier_texts=item.get("carrier_texts", [carrier_text] if carrier_text else []),
                phonetic_text=item.get("phonetic_text", None),
                sense_prototype=item.get("sense_prototype") if same_encoder else None,
                space_data=space_data,
            )
            self._localize(entry)
            loaded_entries.append(entry)
        self.entries = loaded_entries
        self.current_step = max(self.current_step, int(data.get("current_step", 0) or 0))
        target_tau = float(self.config.gate_threshold_init)
        self.gate_threshold.data = torch.tensor(target_tau, dtype=torch.float32, device=self.gate_threshold.device)
        n_local = sum(1 for e in self.entries if e.delta_space == self.space)
        logger.info(
            f"✓ Loaded {len(self.entries)} Hopfield Memory entries from {filepath} "
            f"(space={self.space}, {n_local} with local δ*, gate_tau={self.gate_threshold.item():.2f})"
        )


    def delete_entry(self, word: str) -> bool:
        """Delete an entry or entries matching target word from memory."""
        initial_len = len(self.entries)
        target_norm = word.strip().lower()
        self.entries = [
            e for e in self.entries
            if e.word.strip().lower() != target_norm
        ]
        deleted = len(self.entries) < initial_len
        if deleted:
            logger.info(
                f"✓ Hopfield Memory: Deleted correction for '{word}' "
                f"({initial_len - len(self.entries)} removed, {len(self.entries)} remaining)"
            )
        else:
            logger.warning(f"Hopfield Memory: No entry found to delete for word '{word}'")
        return deleted

    def clear(self) -> None:
        """Clear all stored memory entries."""
        self.entries.clear()
        self.current_step = 0
        if os.path.exists("./corrections.pt"):
            try:
                os.remove("./corrections.pt")
            except Exception:
                pass
        logger.info("Hopfield Memory cleared.")

    def get_vocabulary_prompt(self) -> str:
        """Construct an initial_prompt biasing string for Whisper from stored memory words.

        Conditions Whisper's autoregressive decoder on the exact spellings of learned words
        and domain carrier context.
        """
        if not self.entries:
            return ""
        unique_words = []
        seen = set()
        for e in self.entries:
            w = e.word.strip()
            if w and w.lower() not in seen:
                seen.add(w.lower())
                unique_words.append(w)
        if not unique_words:
            return ""

        context_snippets = []
        for e in self.entries:
            text = (e.carrier_text or "").strip()
            if text and len(text) < 150 and text not in context_snippets:
                context_snippets.append(text)
                if len(context_snippets) >= 2:
                    break

        prompt = "Vocabulary and technical terms: " + ", ".join(unique_words) + "."
        if context_snippets:
            prompt += " " + " ".join(context_snippets)
        return prompt[:450]

    def correct_transcript(
        self,
        text: str,
        threshold: float = 0.65,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Post-correct transcribed text using associative Hopfield Memory entries.

        Detects phonetic misspellings, split words, and acoustic variations in the
        transcript and replaces them with their canonical correct spellings while
        preserving grammatical context.

        Args:
            text: Raw transcribed text from Whisper
            threshold: Minimum similarity threshold (default 0.65)

        Returns:
            Tuple of:
                - corrected_text: Text with canonical spellings restored
                - corrections: List of dicts detailing what was corrected
        """
        if not text or not self.entries:
            return text, []

        corrected_text = text
        applied_corrections: List[Dict[str, Any]] = []

        # 1. Phonetic & Compound Alias Substitution (e.g. entry.phonetic_text -> entry.word)
        for entry in self.entries:
            canonical = entry.word.strip()
            if not canonical:
                continue

            patterns_to_check = []
            if entry.phonetic_text and entry.phonetic_text.strip().lower() != canonical.lower():
                patterns_to_check.append(entry.phonetic_text.strip())

            # Hyphen/space split variants
            if '-' in canonical:
                patterns_to_check.append(canonical.replace('-', ' '))
            if re.search(r'[A-Z]', canonical[1:]) or len(canonical) >= 7:
                split_var = re.sub(r'([a-z])([A-Z])', r'\1 \2', canonical)
                if split_var != canonical:
                    patterns_to_check.append(split_var)
            if canonical.lower().startswith('bi') and len(canonical) > 4:
                patterns_to_check.append('by-' + canonical[2:])
                patterns_to_check.append('by ' + canonical[2:])
                patterns_to_check.append('bi ' + canonical[2:])
                patterns_to_check.append('buy ' + canonical[2:])

            for p in set(patterns_to_check):
                regex_p = r'\b' + re.escape(p).replace(r'\ ', r'[\s-]+') + r'\b'
                matches = list(re.finditer(regex_p, corrected_text, flags=re.IGNORECASE))
                for m in reversed(matches):
                    matched_str = m.group(0)
                    if matched_str.strip().lower() != canonical.lower():
                        start, end = m.span()
                        corrected_text = corrected_text[:start] + canonical + corrected_text[end:]
                        applied_corrections.append({
                            "original": matched_str,
                            "corrected": canonical,
                            "confidence": 0.99,
                            "match_type": "phonetic_alias",
                        })

        # 2. Sliding Window Matching (1 to 4 words)
        words_with_spans = [(m.group(0), m.start(), m.end()) for m in re.finditer(r'\b[A-Za-z0-9]+\b', corrected_text)]
        if not words_with_spans:
            return corrected_text, applied_corrections

        replacements = []
        occupied_spans = set()

        for window_size in range(min(4, len(words_with_spans)), 0, -1):
            for i in range(len(words_with_spans) - window_size + 1):
                window = words_with_spans[i:i + window_size]
                span_range = (window[0][1], window[-1][2])

                if any(start < span_range[1] and end > span_range[0] for (start, end) in occupied_spans):
                    continue

                words_list = [w[0] for w in window]

                best_entry = None
                best_score = 0.0
                best_match_type = ""
                best_target_span = span_range
                best_target_text = corrected_text[span_range[0]:span_range[1]]

                for entry in self.entries:
                    canonical = entry.word.strip()
                    if not canonical:
                        continue

                    # Guard 1: If canonical is ALREADY present in this window, skip
                    if canonical.lower() in [w.lower() for w in words_list]:
                        continue

                    canon_words = set(re.findall(r'\b\w+\b', canonical.lower()))

                    # Guard 2: Strip leading & trailing grammatical stopwords
                    trim_start = 0
                    trim_end = len(window)

                    while trim_start < trim_end - 1:
                        w_clean = window[trim_start][0].lower().strip("'-")
                        if w_clean in GRAMMAR_STOPWORDS and w_clean not in canon_words:
                            trim_start += 1
                        else:
                            break

                    while trim_end > trim_start + 1:
                        w_clean = window[trim_end - 1][0].lower().strip("'-")
                        if w_clean in GRAMMAR_STOPWORDS and w_clean not in canon_words:
                            trim_end -= 1
                        else:
                            break

                    trimmed_window = window[trim_start:trim_end]
                    trimmed_span = (trimmed_window[0][1], trimmed_window[-1][2])
                    trimmed_text = corrected_text[trimmed_span[0]:trimmed_span[1]]

                    full_text_span = span_range
                    full_text = corrected_text[full_text_span[0]:full_text_span[1]]

                    # Hashes
                    h_canon = broad_acoustic_hash(canonical)
                    h_phon = broad_acoustic_hash(entry.phonetic_text or "")

                    # Evaluate trimmed candidate
                    h_trim = broad_acoustic_hash(trimmed_text)
                    lev_trim = levenshtein_ratio(trimmed_text, canonical)
                    if entry.phonetic_text:
                        lev_trim = max(lev_trim, levenshtein_ratio(trimmed_text, entry.phonetic_text))

                    # Evaluate full window candidate (for acoustic splits like "an eventamab" -> "amivantamab")
                    h_full = broad_acoustic_hash(full_text)
                    lev_full = levenshtein_ratio(full_text, canonical)
                    if entry.phonetic_text:
                        lev_full = max(lev_full, levenshtein_ratio(full_text, entry.phonetic_text))

                    cand_span = trimmed_span
                    cand_text = trimmed_text
                    score = 0.0
                    match_type = "acoustic_hash"

                    clean_cand = re.sub(r'[^a-z0-9]', '', cand_text.lower())
                    clean_full = re.sub(r'[^a-z0-9]', '', full_text.lower())
                    clean_canon = re.sub(r'[^a-z0-9]', '', canonical.lower())

                    ratio_trim = min(len(clean_cand), len(clean_canon)) / max(len(clean_cand), len(clean_canon), 1)
                    ratio_full = min(len(clean_full), len(clean_canon)) / max(len(clean_full), len(clean_canon), 1)

                    if h_trim and (h_trim == h_canon or (h_phon and h_trim == h_phon)) and ratio_trim >= 0.70:
                        score = 0.95
                        match_type = "acoustic_hash"
                        cand_span = trimmed_span
                        cand_text = trimmed_text
                    elif h_full and (h_full == h_canon or (h_phon and h_full == h_phon)) and ratio_full >= 0.70:
                        score = 0.95
                        match_type = "acoustic_split_hash"
                        cand_span = full_text_span
                        cand_text = full_text
                    elif lev_trim >= 0.75 and ratio_trim >= 0.70:
                        score = lev_trim
                        match_type = "phonetic_levenshtein"
                        cand_span = trimmed_span
                        cand_text = trimmed_text
                    elif lev_full >= 0.80 and lev_full > lev_trim and ratio_full >= 0.70:
                        score = lev_full
                        match_type = "phonetic_levenshtein_split"
                        cand_span = full_text_span
                        cand_text = full_text

                    # Guard 3: If canonical is ALREADY present in the immediate context around this span, skip
                    ctx_start = max(0, cand_span[0] - len(canonical))
                    ctx_end = min(len(corrected_text), cand_span[1] + len(canonical))
                    if canonical.lower() in corrected_text[ctx_start:ctx_end].lower():
                        continue

                    if any(start < cand_span[1] and end > cand_span[0] for (start, end) in occupied_spans):
                        continue

                    if cand_text.lower() == canonical.lower():
                        continue

                    if score > best_score:
                        best_score = score
                        best_entry = entry
                        best_match_type = match_type
                        best_target_span = cand_span
                        best_target_text = cand_text

                if best_entry is not None and best_score >= threshold:
                    canonical = best_entry.word.strip()
                    replacements.append((
                        best_target_span[0],
                        best_target_span[1],
                        best_target_text,
                        canonical,
                        round(best_score, 3),
                        best_match_type
                    ))
                    occupied_spans.add(best_target_span)

        # Apply sliding window replacements from right to left to keep string indices intact
        replacements.sort(key=lambda r: r[0], reverse=True)
        for start, end, orig, canon, score, m_type in replacements:
            corrected_text = corrected_text[:start] + canon + corrected_text[end:]
            applied_corrections.append({
                "original": orig,
                "corrected": canon,
                "confidence": score,
                "match_type": m_type,
            })

        return corrected_text, applied_corrections

