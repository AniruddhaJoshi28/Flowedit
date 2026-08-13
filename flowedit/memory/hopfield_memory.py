"""
Modern Hopfield Network Memory — Stage 3 of FlowEdit.

Paper Section 3.2 (Stage 3: Associative Memory via Hopfield Networks):
    "Each finalized correction writes a key-value pair:
        K_i = pool(c_I),  V_i = pool(δ*_I)
    where pool averages over the corrected token span. Retrieval follows
    the Modern Hopfield update:
        Mem(Q) = softmax(β·Q·Kᵀ)·V,  β = 1/√d"

This module provides:
    1. Content-addressable episodic memory for pronunciation corrections
    2. Key-value storage with deduplication (cosine sim > 0.95 → EMA merge)
    3. LRU pruning for bounded memory capacity
    4. Context-conditioned keys for homograph disambiguation
    5. Serialization for persistence across sessions
"""

import torch
import torch.nn.functional as F
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field, asdict
from datetime import datetime


from flowedit.config import MemoryConfig

logger = logging.getLogger(__name__)



@dataclass
class PronunciationCorrection:
    """Rich schema-versioned metadata for stored pronunciation corrections."""
    canonical_text: str
    normalized_text: str
    language: str = "en"
    native_script: Optional[str] = None
    phonemes: Tuple[str, ...] = ()
    syllables: Tuple[str, ...] = ()
    left_context: Optional[str] = None
    right_context: Optional[str] = None
    backbone: str = "f5tts"
    model_version: str = "1.0"
    speaker_id: Optional[str] = None
    correction_type: str = "phonetic_alias"
    confidence: float = 1.0
    verified_audio_path: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    preprocessing_version: str = "v1.0"
    correction_strategy_version: str = "v1.0"
    text_processor_version: str = "v1.0"
    embedding_schema_version: str = "v1.0"



class HopfieldMemory:
    """Modern Hopfield Network memory for storing pronunciation corrections.

    This is the "episodic memory" from the paper that gives FlowEdit its
    lifelong learning capability. Corrections are stored as key-value pairs
    and retrieved via soft attention, enabling:
    - Persistent corrections across sessions
    - Fuzzy morphological matching (e.g., "Linux" → "Linux's")
    - Zero forgetting (corrections are external to the model)

    Memory Capacity (paper Section 4.5):
        - Stable up to M=500 corrections (PER increase < 0.2%)
        - At M=1000, softmax dilution raises PER ~0.6%
        - At M=10k, retrieval latency ~112ms (still below perceptibility)
        - Recommend partitioning by domain/language for M>5k
    """

    def __init__(self, dim: int, config: Optional[MemoryConfig] = None):
        """Initialize Hopfield memory.

        Args:
            dim: Embedding dimension (must match backbone embedding_dim)
            config: Memory configuration
        """
        self.dim = dim
        self.config = config or MemoryConfig()

        # Memory storage
        self.keys: List[torch.Tensor] = []      # K_i = pool(c_I)
        self.values: List[torch.Tensor] = []     # V_i = pool(δ*_I)
        self.metadata: List[Dict] = []           # Word labels, timestamps, etc.
        self.access_times: List[float] = []      # For LRU pruning

        # Hopfield inverse temperature: β = 1/√d
        self.beta = self.config.hopfield_beta or (1.0 / (dim ** 0.5))

        logger.info(
            f"HopfieldMemory initialized: dim={dim}, "
            f"max_entries={self.config.max_entries}, "
            f"β={self.beta:.4f}"
        )

    @property
    def size(self) -> int:
        """Number of stored corrections."""
        return len(self.keys)

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        word: str = "",
        context_embeddings: Optional[torch.Tensor] = None,
        target_index_in_context: Optional[int] = None,
    ) -> int:
        """Store a pronunciation correction in memory.

        Paper Eq. 5:
            K_i = pool(c_I)   — average text embedding of corrected tokens
            V_i = pool(δ*_I)  — average perturbation vector

        Deduplication (paper Section 3.2):
            "cosine similarities > 0.95 trigger exponential moving average
            updates instead of new insertions"

        Args:
            key: Key vector (pooled text embedding) [1, dim] or [dim]
            value: Value vector (pooled perturbation) [1, dim] or [dim]
            word: The corrected word (for logging and metadata)
            context_embeddings: Optional surrounding token embeddings for
                               homograph disambiguation [n_context, dim]
            target_index_in_context: Index of the target word in context_embeddings

        Returns:
            Memory index where the correction was stored
        """
        # Ensure correct shape
        key = key.detach().squeeze()
        value = value.detach().squeeze()

        if key.dim() != 1 or key.shape[0] != self.dim:
            raise ValueError(
                f"Key shape mismatch: expected [{self.dim}], got {key.shape}"
            )

        if context_embeddings is not None:
            key = self._apply_context_conditioning(
                key=key,
                context_embeddings=context_embeddings,
                target_index_in_context=target_index_in_context
            )

        # L2-normalize key (paper: "Queries and keys are L2-normalized")
        key = F.normalize(key, dim=0)

        # Check for duplicate — deduplication via cosine similarity
        dedup_idx = self._find_duplicate(key)

        if dedup_idx is not None:
            # EMA update instead of new insertion
            alpha = self.config.dedup_ema_decay
            
            # Ensure key and value are on the same device as the stored keys (usually CPU)
            key_stored = key.to(self.keys[dedup_idx].device)
            value_stored = value.to(self.values[dedup_idx].device)
            
            self.keys[dedup_idx] = alpha * self.keys[dedup_idx] + (1 - alpha) * key_stored
            self.keys[dedup_idx] = F.normalize(self.keys[dedup_idx], dim=0)
            
            if self.values[dedup_idx].shape == value_stored.shape:
                self.values[dedup_idx] = alpha * self.values[dedup_idx] + (1 - alpha) * value_stored
            else:
                # If sequence lengths mismatch (e.g. different Whisper alignments), overwrite with the new one
                self.values[dedup_idx] = value_stored
                
            self.access_times[dedup_idx] = time.time()
            self.metadata[dedup_idx]["update_count"] = (
                self.metadata[dedup_idx].get("update_count", 1) + 1
            )

            logger.info(
                f"Updated existing correction for '{word}' at index {dedup_idx} "
                f"(EMA merge, count: {self.metadata[dedup_idx]['update_count']})"
            )
            return dedup_idx

        # LRU pruning if at capacity
        if self.size >= self.config.max_entries:
            self._prune_lru()

        # Insert new correction
        self.keys.append(key.cpu())
        self.values.append(value.cpu())
        self.access_times.append(time.time())
        self.metadata.append({
            "word": word,
            "created_at": time.time(),
            "update_count": 1,
        })

        idx = self.size - 1
        logger.info(
            f"Stored new correction for '{word}' at index {idx} "
            f"(memory: {self.size}/{self.config.max_entries})"
        )

        return idx

    def retrieve(
        self,
        query: torch.Tensor,
        context_embeddings: Optional[torch.Tensor] = None,
        target_index_in_context: Optional[int] = None,
        backbone: Optional[str] = None,
        model_version: Optional[str] = None,
        embedding_schema_version: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve corrections via Modern Hopfield update with backbone & version isolation."""
        

        if self.is_empty:
            if query.dim() == 1:
                return torch.zeros(self.dim, device=query.device), torch.tensor(-1.0)
            return (
                torch.zeros(query.shape[0], self.dim, device=query.device),
                torch.full((query.shape[0],), -1.0, device=query.device),
            )

        # Filter indices by backbone, model_version, and embedding_schema_version
        valid_indices = []
        for i, meta in enumerate(self.metadata):
            match = True
            if backbone and meta.get("backbone") != backbone:
                match = False
            if model_version and meta.get("model_version") != model_version:
                match = False
            if embedding_schema_version and meta.get("embedding_schema_version") != embedding_schema_version:
                match = False
            if match:
                valid_indices.append(i)

        if not valid_indices:
            if query.dim() == 1:
                return torch.zeros(self.dim, device=query.device), torch.tensor(-1.0)
            return (
                torch.zeros(query.shape[0], self.dim, device=query.device),
                torch.full((query.shape[0],), -1.0, device=query.device),
            )

        # Stack filtered keys into matrix
        filtered_keys = [self.keys[i] for i in valid_indices]
        filtered_values = [self.values[i] for i in valid_indices]

        if context_embeddings is not None:
            query = self._apply_context_conditioning(
                key=query.squeeze(0) if query.dim() > 1 else query,
                context_embeddings=context_embeddings,
                target_index_in_context=target_index_in_context
            )

        K = torch.stack(filtered_keys).to(device=query.device, dtype=query.dtype)  # [M_filtered, d]

        # Ensure query shape
        if query.dim() == 1:
            query = query.unsqueeze(0)  # [1, d]

        # L2-normalize query
        Q = F.normalize(query, dim=-1)

        # Unscaled cosine similarities
        cosine_sims = Q @ K.T
        
        # Max similarity scores
        max_similarities = cosine_sims.max(dim=-1).values  # [batch]
        max_indices = cosine_sims.argmax(dim=-1) # [batch]

        # Compute attention weights
        logits = self.beta * cosine_sims
        weights = F.softmax(logits, dim=-1)

        V = torch.stack(filtered_values).to(device=query.device, dtype=query.dtype) # [M, d]
        retrieved_sequence = (weights @ V).squeeze(0)  # [d]
        
        # Update access times using weights
        self._update_access_times(weights.squeeze(0))


        return retrieved_sequence, max_similarities.squeeze(0)

    def retrieve_batch(
        self,
        queries: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch retrieval for all tokens in a sequence.

        Args:
            queries: Query matrix [seq_len, dim]

        Returns:
            Tuple of:
            - retrieved_values: [seq_len, dim]
            - max_similarities: [seq_len]
        """
        if self.is_empty:
            return (
                torch.zeros_like(queries),
                torch.full((queries.shape[0],), -1.0, device=queries.device),
            )

        K = torch.stack(self.keys).to(device=queries.device, dtype=queries.dtype)  # [M, d]
        pooled_values = [v.mean(dim=0) if v.dim() > 1 else v for v in self.values]
        V = torch.stack(pooled_values).to(device=queries.device, dtype=queries.dtype)  # [M, d]

        Q = F.normalize(queries, dim=-1)  # [seq_len, d]

        # Unscaled cosine similarities for the gate
        cosine_sims = Q @ K.T  # [seq_len, M]
        max_sims = cosine_sims.max(dim=-1).values  # [seq_len]

        # Scaled logits for softmax attention
        logits = self.beta * cosine_sims  # [seq_len, M]

        weights = F.softmax(logits, dim=-1)  # [seq_len, M]
        retrieved = weights @ V  # [seq_len, d]

        return retrieved, max_sims

    def _find_duplicate(self, key: torch.Tensor) -> Optional[int]:
        """Check if a similar key already exists.

        Paper: "cosine similarities > 0.95 trigger EMA updates"

        Returns:
            Index of duplicate entry, or None
        """
        if self.is_empty:
            return None

        K = torch.stack(self.keys).to(device=key.device, dtype=key.dtype)
        similarities = F.cosine_similarity(
            key.unsqueeze(0), K, dim=1
        )

        max_sim, max_idx = similarities.max(dim=0)

        if max_sim.item() > self.config.dedup_cosine_threshold:
            return max_idx.item()

        return None

    def _apply_context_conditioning(
        self,
        key: torch.Tensor,
        context_embeddings: torch.Tensor,
        target_index_in_context: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply context-conditioned key for homograph disambiguation.

        Paper Section 3.2: "For homograph disambiguation (e.g., 'bass' fish
        vs. 'bass' guitar), FlowEdit utilizes context-conditioned keys by
        taking a Gaussian-weighted average of surrounding text embeddings
        within a window of ±3 tokens."

        Args:
            key: The raw key vector [dim]
            context_embeddings: Surrounding token embeddings [n_tokens, dim]
            target_index_in_context: The index of the target word within the context slice.

        Returns:
            Context-conditioned key [dim]
        """
        window = self.config.context_window
        sigma = self.config.context_sigma

        context_embeddings = context_embeddings.to(device=key.device, dtype=key.dtype)
        n_tokens = context_embeddings.shape[0]
        if target_index_in_context is not None:
            center = target_index_in_context
        else:
            center = n_tokens // 2

        # Gaussian weights centered on the target token
        positions = torch.arange(n_tokens, dtype=torch.float32, device=key.device)
        weights = torch.exp(-((positions - center) ** 2) / (2 * sigma ** 2))
        weights = weights / weights.sum()

        # Weighted average of context embeddings
        context_key = (weights.unsqueeze(1) * context_embeddings).sum(dim=0)

        # Combine original key with context (equal weighting)
        conditioned_key = 0.7 * key + 0.3 * context_key

        return conditioned_key.to(key.dtype)

    def _prune_lru(self) -> None:
        """Prune least-recently-used entries when at capacity.

        Paper Section 3.2: "Memory is bounded via LRU pruning at a
        user-defined budget M_max."
        """
        if self.size <= 1:
            return

        # Find the least recently accessed entry
        oldest_idx = min(range(len(self.access_times)),
                         key=lambda i: self.access_times[i])

        word = self.metadata[oldest_idx].get("word", "unknown")
        logger.info(
            f"LRU pruning: removing correction for '{word}' "
            f"at index {oldest_idx} (last accessed: "
            f"{time.time() - self.access_times[oldest_idx]:.0f}s ago)"
        )

        del self.keys[oldest_idx]
        del self.values[oldest_idx]
        del self.metadata[oldest_idx]
        del self.access_times[oldest_idx]

    def _update_access_times(self, weights: torch.Tensor) -> None:
        """Update access timestamps based on attention weights.

        Entries with high attention weight are marked as recently accessed.
        """
        current_time = time.time()
        if weights.dim() == 2:
            max_weights = weights.max(dim=0).values  # [M]
        elif weights.dim() == 1:
            max_weights = weights  # [M]
        else:
            return

        for idx in range(min(self.size, max_weights.shape[0])):
            if max_weights[idx].item() > 0.1:
                self.access_times[idx] = current_time

    def save(self, path: str) -> None:
        """Save memory to disk for persistence across sessions.

        Saves as a dictionary containing keys, values, and metadata.

        Args:
            path: File path (typically .pt extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "dim": self.dim,
            "beta": self.beta,
            "keys": [k.cpu() for k in self.keys],
            "values": [v.cpu() for v in self.values],
            "metadata": self.metadata,
            "access_times": self.access_times,
            "config": {
                "max_entries": self.config.max_entries,
                "dedup_cosine_threshold": self.config.dedup_cosine_threshold,
                "gate_threshold_init": self.config.gate_threshold_init,
            },
        }

        torch.save(save_dict, str(path))
        logger.info(f"Memory saved to {path} ({self.size} corrections)")

    def load(self, path: str) -> None:
        """Load memory from disk.

        Args:
            path: File path to load from

        Raises:
            FileNotFoundError: If path doesn't exist
            ValueError: If dimension mismatch
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Memory file not found: {path}")

        save_dict = torch.load(str(path), map_location="cpu", weights_only=False)

        if save_dict["dim"] != self.dim:
            raise ValueError(
                f"Dimension mismatch: memory has dim={save_dict['dim']}, "
                f"but current config expects dim={self.dim}"
            )

        # Restore keys and values
        raw_keys = save_dict.get("keys", [])
        raw_values = save_dict.get("values", [])

        if isinstance(raw_keys, torch.Tensor):
            self.keys = list(raw_keys) if raw_keys.numel() > 0 else []
        else:
            self.keys = list(raw_keys)

        if isinstance(raw_values, torch.Tensor):
            self.values = list(raw_values) if raw_values.numel() > 0 else []
        else:
            self.values = list(raw_values)

        self.metadata = save_dict.get("metadata", [{}] * len(self.keys))
        self.access_times = save_dict.get(
            "access_times",
            [time.time()] * len(self.keys)
        )
        self.beta = save_dict.get("beta", self.beta)

        logger.info(f"Memory loaded from {path} ({self.size} corrections)")

    def list_corrections(self) -> List[Dict]:
        """List all stored corrections with metadata.

        Returns:
            List of dicts with word, index, and metadata
        """
        corrections = []
        for idx in range(self.size):
            corrections.append({
                "index": idx,
                "word": self.metadata[idx].get("word", "unknown"),
                "update_count": self.metadata[idx].get("update_count", 1),
                "created_at": self.metadata[idx].get("created_at", 0),
                "key_norm": torch.norm(self.keys[idx]).item(),
                "value_norm": torch.norm(self.values[idx]).item(),
            })
        return corrections

    def clear(self) -> None:
        """Clear all stored corrections."""
        self.keys.clear()
        self.values.clear()
        self.metadata.clear()
        self.access_times.clear()
        logger.info("Memory cleared")

    def __repr__(self) -> str:
        return (
            f"HopfieldMemory(dim={self.dim}, size={self.size}/"
            f"{self.config.max_entries}, β={self.beta:.4f})"
        )
