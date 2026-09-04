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
    1. Cluster averaging of (K, V, δ*_word) across speakers and multiple corrections.
    2. Contextual homograph disambiguation via localized Gaussian context keys.
    3. Multi-word simultaneous correction retrieval across full sentences.
    4. Exact BPE token-span localization for precision delta injection.
"""

import os
import re
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

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


@dataclass
class MemoryEntry:
    """A single pronunciation correction entry (or averaged cluster) in Modern Hopfield Memory."""
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


class HopfieldMemory(nn.Module):
    """Modern Continuous Hopfield Associative Memory for Lifelong Pronunciation Adaptation."""

    def __init__(self, config: Optional[MemoryConfig] = None, embedding_dim: int = 512):
        super().__init__()
        self.config = config or MemoryConfig()
        self.d = embedding_dim
        self.beta = self.config.hopfield_beta or (1.0 / math.sqrt(self.d))
        self.max_entries = getattr(self.config, "max_entries", 500)
        self.dedup_threshold = getattr(self.config, "dedup_cosine_threshold", 0.72)
        self.homograph_threshold = getattr(self.config, "homograph_sim_threshold", 0.72)
        self.dedup_ema = getattr(self.config, "dedup_ema_decay", 0.90)
        self.context_window = getattr(self.config, "context_window", 3)
        self.context_char_radius = getattr(self.config, "context_char_radius", 12)
        self.context_sigma = getattr(self.config, "context_sigma", 5.0)

        # Contextual averaging weights for memory key computation
        # K_i = α · word_emb + β · local_context + γ · global_context
        self.context_key_word_weight = getattr(self.config, "context_key_word_weight", 0.70)
        self.context_key_local_weight = getattr(self.config, "context_key_local_weight", 0.20)
        self.context_key_global_weight = getattr(self.config, "context_key_global_weight", 0.10)

        # Gate threshold τ (Paper Section 3.2: τ ≈ 7.0 for precise 0.70 cosine similarity homograph gating)
        self.gate_threshold = nn.Parameter(
            torch.tensor(float(getattr(self.config, "gate_threshold_init", 7.0)), dtype=torch.float32)
        )

        self.entries: List[MemoryEntry] = []
        self.current_step = 0

    @property
    def num_entries(self) -> int:
        return len(self.entries)

    def is_empty(self) -> bool:
        return len(self.entries) == 0

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
    ) -> Tuple[int, str]:
        """Store or average a correction in Modern Hopfield Memory (Paper Section 3.2).

        Sense-Level Canonical Averaging Logic:
        - If an existing entry matches the word and has contextual cosine similarity >= homograph_threshold (0.65):
            Updates keys K, pooled values V, and per-character word perturbations δ*_word using an Exponential Moving Average (EMA).
            This eliminates sentence-specific neighbor noise and accumulates a stable, generalized correction vector for that sense.
        - If the context similarity < homograph_threshold:
            Stores as a distinct contextual homograph cluster (e.g. "lead" metal vs "lead" leader).
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

        # Check for deduplication / cluster averaging vs contextual homographs
        if self.entries:
            word_entries = [
                (i, e) for i, e in enumerate(self.entries)
                if e.word.strip().lower() == word.strip().lower()
            ]

            if word_entries:
                best_sim = -1.0
                best_idx = -1
                for idx, entry in word_entries:
                    sim = torch.dot(key_norm, F.normalize(entry.key.cpu(), p=2, dim=-1)).item()
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = idx

                if best_sim >= self.homograph_threshold and best_idx >= 0:
                    # ─── CANONICAL SENSE-LEVEL EMA AVERAGE (K, V, word_delta) ───
                    matched = self.entries[best_idx]
                    
                    # For early corrections / re-corrections, prioritize the new reference audio
                    if matched.access_count <= 2:
                        decay = 0.20
                        alpha = 0.80
                    else:
                        decay = float(getattr(self.config, "dedup_ema_decay", 0.70))
                        alpha = 1.0 - decay

                    # 1. Key EMA average (normalized)
                    matched_key = matched.key.cpu()
                    avg_key = matched_key * decay + key_norm * alpha
                    matched.key = F.normalize(avg_key, p=2, dim=-1)

                    # 2. Value EMA average
                    matched_val = matched.value.cpu()
                    avg_val = matched_val * decay + val_1d * alpha
                    matched.value = avg_val

                    # 3. Canonical word_delta EMA average (favor new correction)
                    if cur_word_delta is not None:
                        if matched.word_delta is not None and matched.word_delta.shape == cur_word_delta.shape:
                            matched.word_delta = matched.word_delta * decay + cur_word_delta * alpha
                        else:
                            matched.word_delta = cur_word_delta

                    matched.access_count += 1
                    matched.last_access_step = self.current_step
                    matched.metadata.update(metadata or {})
                    if stored_full_delta is not None:
                        matched.full_delta = stored_full_delta

                    if phonetic_text:
                        matched.phonetic_text = phonetic_text

                    if carrier_text:
                        if not hasattr(matched, "carrier_texts") or matched.carrier_texts is None:
                            matched.carrier_texts = [matched.carrier_text] if matched.carrier_text else []
                        if carrier_text not in matched.carrier_texts:
                            matched.carrier_texts.append(carrier_text)

                    logger.info(
                        f"✓ Hopfield Memory Updated: '{word}' (sense cluster N={matched.access_count}, "
                        f"context sim={best_sim:.4f} ≥ {self.homograph_threshold}, entry={best_idx})"
                    )
                    return best_idx, "averaged"
                else:
                    logger.info(
                        f"✓ Hopfield Memory: Distinct Contextual Homograph for '{word}' "
                        f"(context sim={best_sim:.4f} < {self.homograph_threshold}). Storing as separate sense cluster."
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
            metadata=metadata or {},
            full_delta=stored_full_delta,
            word_delta=cur_word_delta,
            carrier_texts=carrier_list,
            phonetic_text=phonetic_text,
        )
        self.entries.append(entry)
        idx = len(self.entries) - 1
        logger.info(f"✓ Hopfield Memory Stored: '{word}' at index {idx} (phonetic='{phonetic_text}', Total Entries: {len(self.entries)})")
        return idx, "inserted"


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
        and accurately injects corrections across all contextually matched target word spans in token space.
        """
        device = query_embeddings.device
        dtype = query_embeddings.dtype
        if query_embeddings.dim() == 3:
            Q = query_embeddings[0]  # [S, d]
        else:
            Q = query_embeddings     # [S, d]

        seq_len, dim = Q.shape

        valid_entries = [e for e in self.entries if e.key.shape[-1] == dim]
        if not valid_entries:
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
        K = torch.stack([e.key.to(device=device, dtype=dtype) for e in valid_entries])   # [M, d]
        V = torch.stack([e.value.to(device=device, dtype=dtype) for e in valid_entries]) # [M, d]

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
        beta_scale = 10.0 if (self.beta < 1.0) else float(self.beta)
        scores = beta_scale * torch.matmul(Q_norm, K_norm.transpose(0, 1))  # [S, M]
        attention_weights = F.softmax(scores, dim=-1)                        # [S, M]
        retrieved_V = torch.matmul(attention_weights, V)                      # [S, d]

        # 5. Similarity Gating (Paper Section 3.2 & Eq. 7)
        tau = self.gate_threshold.to(device=device, dtype=dtype)
        gate_matrix = torch.sigmoid(scores - tau)                            # [S, M]
        max_scores, max_indices = torch.max(scores, dim=-1)                   # [S]
        gate_seq = torch.sigmoid(max_scores - tau)                            # [S]
        gate_seq_clean = torch.where(gate_seq >= 0.50, gate_seq, torch.zeros_like(gate_seq))

        # 6. Multi-Word Contextual Span Matching in Token Space
        matched_spans: List[MatchedWordSpan] = []
        assembled_delta = torch.zeros(1, seq_len, dim, device=device, dtype=dtype)
        gate_map = torch.zeros(1, seq_len, 1, device=device, dtype=dtype)

        if text:
            total_chars = len(text)
            processed_token_spans = set()

            # Tokenize full text if tokenizer is available for exact BPE token boundary resolution
            encoded_tokens = None
            if tokenizer is not None and hasattr(tokenizer, "encode"):
                try:
                    enc = tokenizer.encode(text, lang=language)
                    if isinstance(enc, list) and enc and isinstance(enc[0], list):
                        enc = enc[0]
                    encoded_tokens = enc
                except Exception:
                    encoded_tokens = None

            # Group entries by normalized target word for disambiguating homographs
            word_to_entries: Dict[str, List[Tuple[int, MemoryEntry]]] = {}
            for m_idx, entry in enumerate(valid_entries):
                w_norm = entry.word.strip().lower()
                if w_norm not in word_to_entries:
                    word_to_entries[w_norm] = []
                word_to_entries[w_norm].append((m_idx, entry))

            for w_norm, candidates in word_to_entries.items():
                char_spans = self._find_word_char_spans(text, candidates[0][1].word)
                for c_start, c_end in char_spans:
                    # Resolve token span
                    if encoded_tokens is not None and tokenizer is not None and hasattr(tokenizer, "decode"):
                        # Exact BPE prefix decoding
                        tok_indices = []
                        decoded_prefixes = []
                        for i in range(1, len(encoded_tokens) + 1):
                            try:
                                pref = tokenizer.decode(encoded_tokens[:i])
                                decoded_prefixes.append(len(pref))
                            except Exception:
                                decoded_prefixes.append(decoded_prefixes[-1] if decoded_prefixes else 0)

                        prev_len = 0
                        for idx, curr_len in enumerate(decoded_prefixes):
                            if curr_len > c_start and prev_len < c_end:
                                tok_indices.append(idx)
                            prev_len = curr_len

                        if tok_indices:
                            t_start = max(0, min(tok_indices))
                            t_end = min(seq_len, max(tok_indices) + 1)
                        else:
                            t_start, t_end = self._char_span_to_token_span(c_start, c_end, total_chars, seq_len)
                    else:
                        t_start, t_end = self._char_span_to_token_span(c_start, c_end, total_chars, seq_len)

                    span_start = max(0, t_start)
                    span_end = min(seq_len, max(t_end, span_start + 1))

                    if (span_start, span_end) in processed_token_spans:
                        continue

                    # Search local window around estimated token span
                    search_start = max(0, t_start - 2)
                    search_end = min(seq_len, t_end + 2)

                    # For multiple homograph entries with the same word, pick the one with highest contextual similarity
                    best_m_idx = candidates[0][0]
                    best_entry = candidates[0][1]
                    best_s_occ = -1e9
                    best_g_occ = 0.0

                    for m_idx, entry in candidates:
                        s_cand = scores[search_start:search_end, m_idx].max().item()
                        g_cand = gate_matrix[search_start:search_end, m_idx].max().item()
                        if s_cand > best_s_occ:
                            best_s_occ = s_cand
                            best_g_occ = g_cand
                            best_m_idx = m_idx
                            best_entry = entry

                    s_occ = best_s_occ
                    g_occ = best_g_occ
                    m_idx = best_m_idx
                    entry = best_entry

                    # Gate threshold check: for exact word matches in carrier text, activate injection
                    if g_occ >= 0.20 or s_occ >= 3.0 or (entry.word.strip().lower() in text.lower()):
                        processed_token_spans.add((span_start, span_end))

                        matched_spans.append(
                            MatchedWordSpan(
                                word=entry.word,
                                start_pos=span_start,
                                end_pos=span_end,
                                entry_index=m_idx,
                                score=s_occ,
                                gate_value=1.0,
                                carrier_text=entry.carrier_text,
                                phonetic_text=entry.phonetic_text,
                            )
                        )

                        # Place word perturbation into assembled delta
                        w_delta = entry.word_delta if entry.word_delta is not None else entry.value.unsqueeze(0).unsqueeze(0)
                        w_delta = w_delta.to(device=device, dtype=dtype)
                        src_delta = w_delta[0] if w_delta.dim() == 3 else w_delta
                        span_len = span_end - span_start

                        if src_delta.shape[0] != span_len:
                            if span_len == 1:
                                aligned_delta = src_delta.mean(dim=0, keepdim=True)
                            elif src_delta.shape[0] == 1:
                                aligned_delta = src_delta.repeat(span_len, 1)
                            else:
                                # Linear interpolation so phonemes stretch/contract smoothly across subwords
                                s_3d = src_delta.unsqueeze(0).transpose(1, 2)  # [1, d, L_src]
                                aligned_delta = F.interpolate(s_3d, size=span_len, mode='linear', align_corners=True).transpose(1, 2).squeeze(0)  # [span_len, d]
                        else:
                            aligned_delta = src_delta

                        for p in range(span_len):
                            assembled_delta[0, span_start + p, :] += aligned_delta[p, :]
                            gate_map[0, span_start + p, 0] = 1.0

        # 7. Fallback: If no word spans matched, do not inject arbitrary perturbations
        if not matched_spans:
            assembled_delta = torch.zeros(1, seq_len, dim, device=device, dtype=dtype)
            gate_map = torch.zeros(1, seq_len, 1, device=device, dtype=dtype)

        is_active = (len(matched_spans) > 0) or (gate_map.max().item() > 0.10)

        # Diagnostic summary
        top_matches = []
        for span in matched_spans:
            top_matches.append((span.word, span.score))

        best_full_delta = None
        best_word = None
        best_token_indices = None
        best_carrier = None

        if matched_spans:
            top_span = max(matched_spans, key=lambda s: s.score)
            best_entry = valid_entries[top_span.entry_index]
            best_full_delta = best_entry.full_delta
            best_word = best_entry.word
            best_token_indices = best_entry.token_indices
            best_carrier = best_entry.carrier_text
            logger.info(
                f"[Hopfield Retrieve] Active word corrections: "
                f"{[(s.word, s.start_pos, f'phonetic={s.phonetic_text}', f'gate={s.gate_value:.2f}') for s in matched_spans]}"
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

    def save(self, filepath: str) -> None:
        """Serialize memory entries to disk."""
        data = {
            "entries": [
                {
                    "key": e.key.cpu(),
                    "value": e.value.cpu(),
                    "word": e.word,
                    "carrier_text": e.carrier_text,
                    "carrier_texts": getattr(e, "carrier_texts", [e.carrier_text] if e.carrier_text else []),
                    "token_indices": e.token_indices,
                    "language": e.language,
                    "access_count": e.access_count,
                    "last_access_step": e.last_access_step,
                    "metadata": e.metadata,
                    "full_delta": e.full_delta.cpu() if e.full_delta is not None else None,
                    "word_delta": getattr(e, "word_delta", None).cpu() if getattr(e, "word_delta", None) is not None else None,
                    "phonetic_text": getattr(e, "phonetic_text", None),
                }
                for e in self.entries
            ],
            "current_step": self.current_step,
            "gate_threshold": self.gate_threshold.data.cpu(),
        }
        torch.save(data, filepath)
        logger.info(f"✓ Saved {len(self.entries)} Hopfield Memory entries to {filepath}")

    def load(self, filepath: str) -> None:
        """Load serialized memory entries from disk with backward compatibility."""
        if not os.path.exists(filepath):
            return
        try:
            data = torch.load(filepath, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning(f"Could not load memory file {filepath}: {e}")
            return

        loaded_entries = []
        for item in data.get("entries", []):
            k = item.get("key")
            if k is None or not isinstance(k, torch.Tensor):
                continue
            if k.shape[-1] != self.d:
                logger.warning(
                    f"Skipping incompatible memory entry '{item.get('word')}' with key dimension {k.shape[-1]} "
                    f"(current model backbone requires dimension {self.d})."
                )
                continue

            word = item.get("word", "")
            carrier_text = item.get("carrier_text", "")
            token_indices = item.get("token_indices", [])
            full_delta = item.get("full_delta", None)
            word_delta = item.get("word_delta", None)
            carrier_texts = item.get("carrier_texts", [carrier_text] if carrier_text else [])
            phonetic_text = item.get("phonetic_text", None)

            # Extract word_delta from legacy full_delta if not present
            if word_delta is None and full_delta is not None and word:
                word_delta = self._extract_word_delta(full_delta, word, carrier_text, token_indices)

            loaded_entries.append(
                MemoryEntry(
                    key=item["key"],
                    value=item["value"],
                    word=word,
                    carrier_text=carrier_text,
                    token_indices=token_indices,
                    language=item.get("language", "en"),
                    access_count=item.get("access_count", 1),
                    last_access_step=item.get("last_access_step", 0),
                    metadata=item.get("metadata", {}),
                    full_delta=full_delta,
                    word_delta=word_delta,
                    carrier_texts=carrier_texts,
                    phonetic_text=phonetic_text,
                )
            )
        self.entries = loaded_entries
        target_tau = float(getattr(self.config, "gate_threshold_init", 4.5))
        self.gate_threshold.data = torch.tensor(target_tau, dtype=torch.float32, device=self.gate_threshold.device)
        logger.info(f"✓ Loaded {len(self.entries)} valid Hopfield Memory entries (d={self.d}, gate_tau={self.gate_threshold.item():.2f}) from {filepath}")


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

