#!/usr/bin/env python3
"""
Rigorous Phase 1 Validation & Architectural Audit of F5-TTS DiT Representations.

Audits:
1. text_embed slicing bug vs true frame pooling vs raw character embeddings.
2. Positional and sequence-length confounding factors.
3. 5-Fold Stratified Cross-Validation on an expanded, position-balanced 80-sentence benchmark with hard semantic negatives.
4. Per-homograph accuracy, majority-class baselines, and 95% Wilson score confidence intervals.
5. Noise stability vs pronunciation discrimination stability (Fisher d-prime).
6. Comparative baselines: Raw Character Lookup, Sentence-Pooled Characters, Corrected text_embed, DiT Blocks (2, 10, 14, 21), and all-MiniLM-L6-v2.
7. Pure empirical evaluation for adapter recommendation.
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_probing")

# ---------------------------------------------------------------------------
# 1. Expanded & Hardened Benchmark Dataset (80 Sentences)
# Balanced 10 per sense (5 standard + 5 hard negatives/distractors with misleading keywords & balanced positions)
# ---------------------------------------------------------------------------

EXPANDED_HARDENED_BENCHMARK = [
    # ========================== LEAD ==========================
    # Sense 0: /lɛd/ (metal / noun) - 10 sentences
    {"word": "lead", "sense_id": 0, "pronunciation": "/lɛd/", "type": "standard",
     "sentence": "The old plumbing pipes were made of toxic lead."},
    {"word": "lead", "sense_id": 0, "pronunciation": "/lɛd/", "type": "standard",
     "sentence": "Workers wore heavy protective lead aprons in the laboratory."},
    {"word": "lead", "sense_id": 0, "pronunciation": "/lɛd/", "type": "standard",
     "sentence": "The vintage paint contained dangerous levels of lead."},
    {"word": "lead", "sense_id": 0, "pronunciation": "/lɛd/", "type": "standard",
     "sentence": "The antique bullets were cast from solid lead."},
    {"word": "lead", "sense_id": 0, "pronunciation": "/lɛd/", "type": "standard",
     "sentence": "Dense lead shields were installed to block the radiation."},
    # Hard Negatives / Distractors (Misleading leadership/guidance keywords & balanced positions)
    {"word": "lead", "sense_id": 0, "pronunciation": "/lɛd/", "type": "hard_negative",
     "sentence": "Lead metal was transported by the team director to the storage bay."}, # position: start
    {"word": "lead", "sense_id": 0, "pronunciation": "/lɛd/", "type": "hard_negative",
     "sentence": "Under executive guidance, the crew safely melted the heavy lead weights."}, # guidance keywords
    {"word": "lead", "sense_id": 0, "pronunciation": "/lɛd/", "type": "hard_negative",
     "sentence": "The committee inspected the lead shielding before the project launch."}, # position: mid, project keywords
    {"word": "lead", "sense_id": 0, "pronunciation": "/lɛd/", "type": "hard_negative",
     "sentence": "To pioneer new medical standards, doctors measured residual lead in the blood."}, # pioneer keywords
    {"word": "lead", "sense_id": 0, "pronunciation": "/lɛd/", "type": "hard_negative",
     "sentence": "Experienced foremen know that ancient pipes are often made of lead."}, # position: end, management keywords

    # Sense 1: /liːd/ (guide / verb/noun) - 10 sentences
    {"word": "lead", "sense_id": 1, "pronunciation": "/liːd/", "type": "standard",
     "sentence": "Priya was appointed as the lead researcher on the project."},
    {"word": "lead", "sense_id": 1, "pronunciation": "/liːd/", "type": "standard",
     "sentence": "He will lead the engineering team through the next launch."},
    {"word": "lead", "sense_id": 1, "pronunciation": "/liːd/", "type": "standard",
     "sentence": "She was chosen to lead the expedition across the mountains."},
    {"word": "lead", "sense_id": 1, "pronunciation": "/liːd/", "type": "standard",
     "sentence": "Effective executives lead by example and inspire confidence."},
    {"word": "lead", "sense_id": 1, "pronunciation": "/liːd/", "type": "standard",
     "sentence": "Who will lead the panel discussion at the conference tomorrow?"},
    # Hard Negatives / Distractors (Misleading metal/chemical/weight keywords & balanced positions)
    {"word": "lead", "sense_id": 1, "pronunciation": "/liːd/", "type": "hard_negative",
     "sentence": "Lead researchers must analyze the toxic chemicals with extreme caution."}, # position: start, toxic/chemicals
    {"word": "lead", "sense_id": 1, "pronunciation": "/liːd/", "type": "hard_negative",
     "sentence": "The chief metallurgist will lead the alloy inspection across the facility."}, # metallurgist/alloy
    {"word": "lead", "sense_id": 1, "pronunciation": "/liːd/", "type": "hard_negative",
     "sentence": "Heavy structural renovations require an experienced architect to lead."}, # position: end, heavy
    {"word": "lead", "sense_id": 1, "pronunciation": "/liːd/", "type": "hard_negative",
     "sentence": "Radiation safety officers frequently lead emergency drills in nuclear plants."}, # radiation safety
    {"word": "lead", "sense_id": 1, "pronunciation": "/liːd/", "type": "hard_negative",
     "sentence": "The miner agreed to lead the survey team through the subterranean tunnels."}, # miner/subterranean

    # ========================== READ ==========================
    # Sense 0: /rɛd/ (past tense) - 10 sentences
    {"word": "read", "sense_id": 0, "pronunciation": "/rɛd/", "type": "standard",
     "sentence": "Yesterday evening she read the entire confidential dossier."},
    {"word": "read", "sense_id": 0, "pronunciation": "/rɛd/", "type": "standard",
     "sentence": "I already read that published article earlier this morning."},
    {"word": "read", "sense_id": 0, "pronunciation": "/rɛd/", "type": "standard",
     "sentence": "The committee carefully read every submitted grant proposal."},
    {"word": "read", "sense_id": 0, "pronunciation": "/rɛd/", "type": "standard",
     "sentence": "He read through the entire technical manual last weekend."},
    {"word": "read", "sense_id": 0, "pronunciation": "/rɛd/", "type": "standard",
     "sentence": "She read the breaking news before anyone else in the office."},
    # Hard Negatives / Distractors (Future/habitual words & balanced positions)
    {"word": "read", "sense_id": 0, "pronunciation": "/rɛd/", "type": "hard_negative",
     "sentence": "Read documents were filed away after the previous audit concluded."}, # position: start
    {"word": "read", "sense_id": 0, "pronunciation": "/rɛd/", "type": "hard_negative",
     "sentence": "Before attending tomorrow's seminar, the student read all assigned chapters."}, # tomorrow keyword
    {"word": "read", "sense_id": 0, "pronunciation": "/rɛd/", "type": "hard_negative",
     "sentence": "The contract was signed once both partners read the fine print."}, # position: mid
    {"word": "read", "sense_id": 0, "pronunciation": "/rɛd/", "type": "hard_negative",
     "sentence": "Although he plans future revisions, the editor read the draft yesterday."}, # future revisions
    {"word": "read", "sense_id": 0, "pronunciation": "/rɛd/", "type": "hard_negative",
     "sentence": "That was the most captivating novel I ever read."}, # position: end

    # Sense 1: /riːd/ (present tense / infinitive) - 10 sentences
    {"word": "read", "sense_id": 1, "pronunciation": "/riːd/", "type": "standard",
     "sentence": "I like to read contemporary poetry before going to sleep."},
    {"word": "read", "sense_id": 1, "pronunciation": "/riːd/", "type": "standard",
     "sentence": "Students must read the assigned textbook chapter every week."},
    {"word": "read", "sense_id": 1, "pronunciation": "/riːd/", "type": "standard",
     "sentence": "Can you please read the instructions aloud to the entire room?"},
    {"word": "read", "sense_id": 1, "pronunciation": "/riːd/", "type": "standard",
     "sentence": "We read scientific journals regularly to stay up to date."},
    {"word": "read", "sense_id": 1, "pronunciation": "/riːd/", "type": "standard",
     "sentence": "You should always read the terms and conditions before signing."},
    # Hard Negatives / Distractors (Past keywords & balanced positions)
    {"word": "read", "sense_id": 1, "pronunciation": "/riːd/", "type": "hard_negative",
     "sentence": "Read instructions carefully when assembling vintage furniture."}, # position: start, vintage keyword
    {"word": "read", "sense_id": 1, "pronunciation": "/riːd/", "type": "hard_negative",
     "sentence": "Historians study ancient archives and read dusty medieval manuscripts daily."}, # ancient/medieval keywords
    {"word": "read", "sense_id": 1, "pronunciation": "/riːd/", "type": "hard_negative",
     "sentence": "To pass the historical examination, candidates must read original diaries."}, # historical keyword
    {"word": "read", "sense_id": 1, "pronunciation": "/riːd/", "type": "hard_negative",
     "sentence": "Yesterday was chaotic, but today we can calmly read the reports."}, # yesterday keyword
    {"word": "read", "sense_id": 1, "pronunciation": "/riːd/", "type": "hard_negative",
     "sentence": "Ancient stone inscriptions are difficult for modern linguists to read."}, # position: end, ancient keyword

    # ========================== BOW ==========================
    # Sense 0: /boʊ/ (weapon / ribbon / noun) - 10 sentences
    {"word": "bow", "sense_id": 0, "pronunciation": "/boʊ/", "type": "standard",
     "sentence": "The skilled archer aimed his wooden bow with steady hands."},
    {"word": "bow", "sense_id": 0, "pronunciation": "/boʊ/", "type": "standard",
     "sentence": "She tied a decorative silk bow around the birthday present."},
    {"word": "bow", "sense_id": 0, "pronunciation": "/boʊ/", "type": "standard",
     "sentence": "He crafted a recurve bow from seasoned hickory wood."},
    {"word": "bow", "sense_id": 0, "pronunciation": "/boʊ/", "type": "standard",
     "sentence": "A neat velvet bow adorned the collar of her evening dress."},
    {"word": "bow", "sense_id": 0, "pronunciation": "/boʊ/", "type": "standard",
     "sentence": "The hunter inspected his compound bow before entering the woods."},
    # Hard Negatives / Distractors (Performance/reverence keywords & balanced positions)
    {"word": "bow", "sense_id": 0, "pronunciation": "/boʊ/", "type": "hard_negative",
     "sentence": "Bow strings must be waxed prior to performing in the competition."}, # position: start, performance
    {"word": "bow", "sense_id": 0, "pronunciation": "/boʊ/", "type": "hard_negative",
     "sentence": "The violinist gripped the carbon fiber bow before the solemn ceremony."}, # ceremony/solemn
    {"word": "bow", "sense_id": 0, "pronunciation": "/boʊ/", "type": "hard_negative",
     "sentence": "During the formal greeting, the gift with the satin bow was presented."}, # formal greeting keywords
    {"word": "bow", "sense_id": 0, "pronunciation": "/boʊ/", "type": "hard_negative",
     "sentence": "Traditional archers carry a quiver and a lacquered hunting bow."}, # position: end
    {"word": "bow", "sense_id": 0, "pronunciation": "/boʊ/", "type": "hard_negative",
     "sentence": "The martial arts instructor examined the craftsmanship of the wooden bow."}, # martial arts instructor keywords

    # Sense 1: /baʊ/ (gesture / reverence / verb/noun) - 10 sentences
    {"word": "bow", "sense_id": 1, "pronunciation": "/baʊ/", "type": "standard",
     "sentence": "The lead actors took a gracious bow as the audience applauded."},
    {"word": "bow", "sense_id": 1, "pronunciation": "/baʊ/", "type": "standard",
     "sentence": "Visitors must bow respectfully when entering the sacred shrine."},
    {"word": "bow", "sense_id": 1, "pronunciation": "/baʊ/", "type": "standard",
     "sentence": "He gave a polite formal bow to greet the foreign dignitaries."},
    {"word": "bow", "sense_id": 1, "pronunciation": "/baʊ/", "type": "standard",
     "sentence": "The violinist made a humble bow before beginning the concerto."},
    {"word": "bow", "sense_id": 1, "pronunciation": "/baʊ/", "type": "standard",
     "sentence": "Martial artists bow to their opponents before every match."},
    # Hard Negatives / Distractors (Weapon/ribbon/craft keywords & balanced positions)
    {"word": "bow", "sense_id": 1, "pronunciation": "/baʊ/", "type": "hard_negative",
     "sentence": "Bow heads in silent reverence before the fallen timber monument."}, # position: start, timber
    {"word": "bow", "sense_id": 1, "pronunciation": "/baʊ/", "type": "hard_negative",
     "sentence": "The archer stepped forward onto the platform to bow graciously."}, # archer keyword
    {"word": "bow", "sense_id": 1, "pronunciation": "/baʊ/", "type": "hard_negative",
     "sentence": "Decorated with ribbons and flowers, performers will bow together."}, # ribbons/flowers keywords
    {"word": "bow", "sense_id": 1, "pronunciation": "/baʊ/", "type": "hard_negative",
     "sentence": "Before competing with recurve weapons, rival athletes always bow."}, # position: end, recurve weapons
    {"word": "bow", "sense_id": 1, "pronunciation": "/baʊ/", "type": "hard_negative",
     "sentence": "Diplomats dressed in velvet attire gave a courtly bow to the king."}, # velvet attire keywords

    # ========================== TEAR ==========================
    # Sense 0: /tɪər/ (crying / drop / noun/verb) - 10 sentences
    {"word": "tear", "sense_id": 0, "pronunciation": "/tɪər/", "type": "standard",
     "sentence": "A single warm tear rolled down her cheek during the ceremony."},
    {"word": "tear", "sense_id": 0, "pronunciation": "/tɪər/", "type": "standard",
     "sentence": "He wiped away a tear after hearing the emotional family story."},
    {"word": "tear", "sense_id": 0, "pronunciation": "/tɪər/", "type": "standard",
     "sentence": "Her eyes began to tear from the pungent smoke of the campfire."},
    {"word": "tear", "sense_id": 0, "pronunciation": "/tɪər/", "type": "standard",
     "sentence": "She blinked back a tear to stay composed during the difficult interview."},
    {"word": "tear", "sense_id": 0, "pronunciation": "/tɪər/", "type": "standard",
     "sentence": "No one shed a tear when the corrupt mayor resigned from office."},
    # Hard Negatives / Distractors (Fabric/sharp/damage keywords & balanced positions)
    {"word": "tear", "sense_id": 0, "pronunciation": "/tɪər/", "type": "hard_negative",
     "sentence": "Tear drops stained the edge of the frayed fabric on her lap."}, # position: start, frayed fabric
    {"word": "tear", "sense_id": 0, "pronunciation": "/tɪər/", "type": "hard_negative",
     "sentence": "While cutting the damaged cloth, the tailor blinked away a tear."}, # position: end, cutting/cloth
    {"word": "tear", "sense_id": 0, "pronunciation": "/tɪər/", "type": "hard_negative",
     "sentence": "Sharp chemical odors caused a sudden tear in the researcher's eye."}, # sharp keyword
    {"word": "tear", "sense_id": 0, "pronunciation": "/tɪər/", "type": "hard_negative",
     "sentence": "She noticed a tear form on his eyelid while repairing the jacket."}, # repairing jacket keywords
    {"word": "tear", "sense_id": 0, "pronunciation": "/tɪər/", "type": "hard_negative",
     "sentence": "Deep emotional pain brought a silent tear to her tired eyes."}, # position: mid

    # Sense 1: /tɛər/ (rip / fracture / verb/noun) - 10 sentences
    {"word": "tear", "sense_id": 1, "pronunciation": "/tɛər/", "type": "standard",
     "sentence": "The seamstress will tear the vintage fabric along the woven grain."},
    {"word": "tear", "sense_id": 1, "pronunciation": "/tɛər/", "type": "standard",
     "sentence": "The tailor discovered a jagged tear along the jacket sleeve."},
    {"word": "tear", "sense_id": 1, "pronunciation": "/tɛər/", "type": "standard",
     "sentence": "Do not tear the receipt if you plan to return the merchandise."},
    {"word": "tear", "sense_id": 1, "pronunciation": "/tɛər/", "type": "standard",
     "sentence": "A sudden violent motion can tear a knee ligament during sports."},
    {"word": "tear", "sense_id": 1, "pronunciation": "/tɛər/", "type": "standard",
     "sentence": "Strong wind gusts began to tear the cloth canopy apart."},
    # Hard Negatives / Distractors (Emotional/weeping/sad keywords & balanced positions)
    {"word": "tear", "sense_id": 1, "pronunciation": "/tɛər/", "type": "hard_negative",
     "sentence": "Tear the old love letters into scraps during this emotional moment."}, # position: start, emotional/letters
    {"word": "tear", "sense_id": 1, "pronunciation": "/tɛər/", "type": "hard_negative",
     "sentence": "In an angry fit of grief and weeping, he began to tear the photos."}, # grief/weeping keywords
    {"word": "tear", "sense_id": 1, "pronunciation": "/tɛər/", "type": "hard_negative",
     "sentence": "A sad incident caused the fragile paper banner to tear."}, # position: end, sad incident
    {"word": "tear", "sense_id": 1, "pronunciation": "/tɛər/", "type": "hard_negative",
     "sentence": "The grieving seamstress accidentally caused a tear in the silk dress."}, # grieving keyword
    {"word": "tear", "sense_id": 1, "pronunciation": "/tɛər/", "type": "hard_negative",
     "sentence": "Rough handling will easily tear the fragile memorial manuscript."}, # memorial keyword
]


# ---------------------------------------------------------------------------
# 2. Alignment & Feature Extraction Utilities
# ---------------------------------------------------------------------------

def locate_word_character_span(text: str, word: str) -> Tuple[int, int]:
    """Finds exact character index [start, end) of whole target word."""
    pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        pattern_flex = re.compile(r"\b" + re.escape(word) + r"(?:s|ed|ing|'s)?\b", re.IGNORECASE)
        match = pattern_flex.search(text)
    if not match:
        raise ValueError(f"Word '{word}' not found in sentence: '{text}'")
    return match.start(), match.end()


def compute_f5_char_to_frame_mapping(text_len: int, audio_len: int) -> List[int]:
    """Replicates exact F5-TTS late average upsampling index map (dit.py:55-84)."""
    if text_len <= 0 or audio_len <= 0:
        return []
    base_repeat = audio_len // text_len
    remainder = audio_len % text_len
    indices = []
    for j in range(text_len):
        repeat_count = base_repeat + (1 if j >= text_len - remainder else 0)
        indices.extend([j] * repeat_count)
    return indices[:audio_len]


def pool_target_frames(
    hidden_states: torch.Tensor,
    char_span: Tuple[int, int],
    char_to_frame_map: List[int],
) -> torch.Tensor:
    """Mean-pools DiT frame activations corresponding to target word characters."""
    cs, ce = char_span
    target_frame_indices = [
        frame_idx
        for frame_idx, c_idx in enumerate(char_to_frame_map)
        if cs <= c_idx < ce
    ]
    if not target_frame_indices:
        n_frames = hidden_states.shape[1]
        text_len = max(1, char_to_frame_map[-1] + 1 if char_to_frame_map else len(char_to_frame_map))
        f_start = int(cs * n_frames / text_len)
        f_end = max(f_start + 1, int(ce * n_frames / text_len))
        target_frame_indices = list(range(f_start, min(f_end, n_frames)))

    selected = hidden_states[0, target_frame_indices, :]
    return selected.mean(dim=0)


# ---------------------------------------------------------------------------
# 3. Statistical Metrics & Confidence Interval Calculation
# ---------------------------------------------------------------------------

def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculates Wilson score 95% confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.95996  # 95% confidence
    p_hat = k / n
    denom = 1.0 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2 * n)) / denom
    margin = (z * math.sqrt((p_hat * (1 - p_hat) / n) + (z ** 2) / (4 * (n ** 2)))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (round(lower, 3), round(upper, 3))


def compute_fisher_dprime(X: np.ndarray, y: np.ndarray) -> float:
    """Computes normalized d-prime (Fisher separability criterion) between binary classes."""
    u_labels = np.unique(y)
    if len(u_labels) != 2:
        return 0.0
    c0 = X[y == u_labels[0]]
    c1 = X[y == u_labels[1]]
    if len(c0) < 2 or len(c1) < 2:
        return 0.0
    mu0, mu1 = np.mean(c0, axis=0), np.mean(c1, axis=0)
    var0 = np.var(c0, axis=0, ddof=1)
    var1 = np.var(c1, axis=0, ddof=1)
    pooled_sd = np.sqrt(0.5 * (np.sum(var0) + np.sum(var1)))
    dist = np.linalg.norm(mu0 - mu1)
    return float(dist / (pooled_sd + 1e-8))


def compute_silhouette(features: np.ndarray, labels: np.ndarray) -> float:
    """Computes silhouette coefficient across classes."""
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    n = len(features)
    diff = features[:, np.newaxis, :] - features[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))

    s_scores = []
    for i in range(n):
        c_i = labels[i]
        same_class = (labels == c_i)
        same_class[i] = False
        if not np.any(same_class):
            continue
        a_i = np.mean(dists[i, same_class])
        b_i = float("inf")
        for other_c in unique_labels:
            if other_c == c_i:
                continue
            other_class = (labels == other_c)
            b_i = min(b_i, np.mean(dists[i, other_class]))
        denom = max(a_i, b_i)
        s_scores.append((b_i - a_i) / denom if denom > 1e-12 else 0.0)
    return float(np.mean(s_scores)) if s_scores else 0.0


def compute_stratified_5fold_cv(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Tuple[float, int, int]:
    """Stratified 5-Fold cross-validation with L2-regularized logistic regression.
    
    Returns:
        (accuracy, total_correct, total_evaluated)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        return 1.0, len(y), len(y)

    n_splits = 5
    indices_by_class = {c: np.where(y == c)[0] for c in unique_labels}
    for c in indices_by_class:
        np.random.shuffle(indices_by_class[c])

    fold_correct = 0
    fold_total = 0

    for f in range(n_splits):
        test_idx = []
        train_idx = []
        for c in unique_labels:
            c_idx = indices_by_class[c]
            fold_size = len(c_idx) // n_splits
            start = f * fold_size
            end = (f + 1) * fold_size if f < n_splits - 1 else len(c_idx)
            test_idx.extend(c_idx[start:end])
            train_idx.extend(np.concatenate([c_idx[:start], c_idx[end:]]))

        X_tr, y_tr = torch.from_numpy(X[train_idx]).float(), torch.from_numpy(y[train_idx]).long()
        X_te, y_te = torch.from_numpy(X[test_idx]).float(), torch.from_numpy(y[test_idx]).long()

        dim = X_tr.shape[1]
        n_classes = len(unique_labels)
        linear = nn.Linear(dim, n_classes)
        opt = torch.optim.Adam(linear.parameters(), lr=0.01, weight_decay=1e-2)
        loss_fn = nn.CrossEntropyLoss()

        linear.train()
        for _ in range(80):
            opt.zero_grad()
            loss = loss_fn(linear(X_tr), y_tr)
            loss.backward()
            opt.step()

        linear.eval()
        with torch.no_grad():
            preds = linear(X_te).argmax(dim=-1)
            correct = (preds == y_te).sum().item()
            fold_correct += correct
            fold_total += len(y_te)

    acc = fold_correct / fold_total if fold_total > 0 else 0.0
    return float(acc), int(fold_correct), int(fold_total)


# ---------------------------------------------------------------------------
# 4. Multi-Layer Activation Probe
# ---------------------------------------------------------------------------

class DiTAuditHookProbe:
    """Non-invasively hooks DiT blocks and text embedding."""

    def __init__(self, dit_model: nn.Module, target_layers: List[int]):
        self.model = dit_model
        self.target_layers = target_layers
        self.hooks = []
        self.activations: Dict[int, torch.Tensor] = {}
        self.text_embed_activation: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        tb = getattr(self.model, "transformer_blocks", None)
        if tb is not None:
            for l_idx in self.target_layers:
                if 0 <= l_idx < len(tb):
                    self.hooks.append(tb[l_idx].register_forward_hook(self._make_block_hook(l_idx)))

        te = getattr(self.model, "text_embed", None)
        if te is not None:
            self.hooks.append(te.register_forward_hook(self._text_embed_hook))

    def _make_block_hook(self, layer_idx: int):
        def hook(module, inp, out):
            self.activations[layer_idx] = out.detach().clone()
        return hook

    def _text_embed_hook(self, module, inp, out):
        self.text_embed_activation = out.detach().clone()

    def clear(self):
        self.activations.clear()
        self.text_embed_activation = None

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.clear()


# ---------------------------------------------------------------------------
# 5. Core Audit Runner
# ---------------------------------------------------------------------------

def run_rigorous_audit(device: str = "cpu", output_json: str = "audit_validation_results.json"):
    logger.info("=" * 80)
    logger.info("PHASE 1 RIGOROUS VALIDATION & REPRODUCIBILITY AUDIT")
    logger.info("=" * 80)
    logger.info(f"Target Homographs: ['lead', 'read', 'bow', 'tear']")
    logger.info(f"Total Sentences:   {len(EXPANDED_HARDENED_BENCHMARK)} (20 per word: 10 standard + 10 hard negatives)")
    logger.info(f"Evaluation Method: Stratified 5-Fold Cross-Validation, Wilson 95% CIs, Fisher d-prime")

    # 1. Load F5-TTS
    from f5_tts.api import F5TTS
    from f5_tts.model.utils import list_str_to_idx
    import site

    logger.info("Loading official pretrained F5-TTS...")
    tts_api = F5TTS(model="F5TTS_Base", device=device)
    model = getattr(tts_api, "ema_model", getattr(tts_api, "model", None))
    dit = getattr(model, "transformer", model)

    vocab_candidates = []
    for sp in site.getsitepackages():
        vocab_candidates.append(os.path.join(sp, "f5_tts", "infer", "examples", "vocab.txt"))
    flowedit_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    vocab_candidates.extend([
        os.path.join(flowedit_root, "models", "f5tts", "f5tts", "F5TTS_Base", "vocab.txt"),
        os.path.join(flowedit_root, "data", "Emilia_ZH_EN_pinyin", "vocab.txt"),
    ])
    char_map = None
    for vc in vocab_candidates:
        if os.path.isfile(vc) and os.path.getsize(vc) > 50:
            with open(vc, "r", encoding="utf-8") as f:
                char_map = {line.strip(): i for i, line in enumerate(f)}
            break
    if not char_map:
        raise RuntimeError("Could not find F5-TTS vocab.txt")

    target_layers = [2, 10, 14, 21]
    probe = DiTAuditHookProbe(dit, target_layers)

    # 2. Load MiniLM Baseline
    logger.info("Loading sentence-transformers/all-MiniLM-L6-v2 contextual baseline...")
    from transformers import AutoTokenizer, AutoModel
    minilm_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    minilm_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    minilm_model.eval()

    target_words = ["lead", "read", "bow", "tear"]

    all_rep_keys = [
        "raw_char_lookup",
        "sentence_char_pool",
        "buggy_text_embed",
        "corrected_text_embed",
        "dit_layer_2",
        "dit_layer_10",
        "dit_layer_14",
        "dit_layer_21",
        "minilm_target",
        "minilm_sentence",
    ]

    features_by_word_and_rep: Dict[str, Dict[str, List[np.ndarray]]] = {
        w: {k: [] for k in all_rep_keys} for w in target_words
    }
    noise_pert_by_word_and_rep: Dict[str, Dict[str, List[List[np.ndarray]]]] = {
        w: {k: [[] for _ in range(20)] for k in all_rep_keys} for w in target_words
    }
    labels_by_word: Dict[str, List[int]] = {w: [] for w in target_words}

    raw_char_table = dit.text_embed.text_embed # nn.Embedding(2546, 512)

    logger.info("Extracting representations across all 80 benchmark sentences...")
    t_start_total = time.time()

    for word in target_words:
        items = [x for x in EXPANDED_HARDENED_BENCHMARK if x["word"] == word]
        logger.info(f"\nProcessing word: '{word}' ({len(items)} sentences)...")

        word_char_ids = torch.tensor([char_map.get(c, 0) + 1 for c in word], dtype=torch.long, device=device)
        static_word_vec = raw_char_table(word_char_ids).mean(dim=0).detach().cpu().numpy()

        for item_idx, item in enumerate(items):
            sent = item["sentence"]
            labels_by_word[word].append(item["sense_id"])
            cs, ce = locate_word_character_span(sent, word)

            # MiniLM Target and Sentence Embeddings
            with torch.no_grad():
                m_enc = minilm_tok(sent, return_offsets_mapping=True, return_tensors="pt").to(device)
                m_out = minilm_model(input_ids=m_enc["input_ids"], attention_mask=m_enc["attention_mask"])
                m_cls = m_out.last_hidden_state[0, 0, :].cpu().numpy()

                offsets = m_enc["offset_mapping"][0].cpu().numpy()
                target_token_indices = [
                    t_idx for t_idx, (t_s, t_e) in enumerate(offsets)
                    if t_s >= cs and t_e <= ce and t_s < t_e
                ]
                if not target_token_indices:
                    target_token_indices = [
                        t_idx for t_idx, (t_s, t_e) in enumerate(offsets)
                        if max(cs, t_s) < min(ce, t_e) and t_s < t_e
                    ]
                if target_token_indices:
                    m_target = m_out.last_hidden_state[0, target_token_indices, :].mean(dim=0).cpu().numpy()
                else:
                    m_target = m_cls

            features_by_word_and_rep[word]["minilm_target"].append(m_target)
            features_by_word_and_rep[word]["minilm_sentence"].append(m_cls)

            # Raw Character Lookup & Sentence Character Pool
            features_by_word_and_rep[word]["raw_char_lookup"].append(static_word_vec)
            sent_char_ids = torch.tensor([char_map.get(c, 0) + 1 for c in sent], dtype=torch.long, device=device)
            sent_char_vec = raw_char_table(sent_char_ids).mean(dim=0).detach().cpu().numpy()
            features_by_word_and_rep[word]["sentence_char_pool"].append(sent_char_vec)

            # F5-TTS Forward Passes (Primary pass + 2 noise perturbation passes)
            text_tensor = list_str_to_idx([sent], char_map).to(device)
            text_len = text_tensor.shape[1]
            est_duration = max(0.5, text_len / 12.5)
            audio_len = int(est_duration * 86.0)
            c2f = compute_f5_char_to_frame_mapping(text_len, audio_len)

            for seed_idx, n_seed in enumerate([42, 142, 242]):
                torch.manual_seed(n_seed)
                probe.clear()
                x_t = torch.randn(1, audio_len, 100, device=device) # t=0.0
                cond_audio = torch.zeros(1, audio_len, 100, device=device)
                t_tensor = torch.tensor([0.0], dtype=torch.float32, device=device)

                with torch.no_grad():
                    _ = dit(x_t, cond_audio, text_tensor, t_tensor)

                act_te = probe.text_embed_activation
                buggy_v = act_te[0, cs:ce, :].mean(dim=0).cpu().numpy()
                corr_v = pool_target_frames(act_te, (cs, ce), c2f).cpu().numpy()

                d2_v = pool_target_frames(probe.activations[2], (cs, ce), c2f).cpu().numpy()
                d10_v = pool_target_frames(probe.activations[10], (cs, ce), c2f).cpu().numpy()
                d14_v = pool_target_frames(probe.activations[14], (cs, ce), c2f).cpu().numpy()
                d21_v = pool_target_frames(probe.activations[21], (cs, ce), c2f).cpu().numpy()

                current_pass_dict = {
                    "buggy_text_embed": buggy_v,
                    "corrected_text_embed": corr_v,
                    "dit_layer_2": d2_v,
                    "dit_layer_10": d10_v,
                    "dit_layer_14": d14_v,
                    "dit_layer_21": d21_v,
                }

                if seed_idx == 0:
                    for k_name, vec in current_pass_dict.items():
                        features_by_word_and_rep[word][k_name].append(vec)

                for k_name, vec in current_pass_dict.items():
                    noise_pert_by_word_and_rep[word][k_name][item_idx].append(vec)

            for rep_k in ["raw_char_lookup", "sentence_char_pool", "minilm_target", "minilm_sentence"]:
                v_rep = features_by_word_and_rep[word][rep_k][-1]
                noise_pert_by_word_and_rep[word][rep_k][item_idx] = [v_rep, v_rep, v_rep]

    probe.close()
    elapsed_total = time.time() - t_start_total
    logger.info(f"✓ Feature extraction complete in {elapsed_total:.1f}s.")

    # 3. Compute Metrics Across Representations and Words
    results = {
        "metadata": {
            "num_sentences": len(EXPANDED_HARDENED_BENCHMARK),
            "words": target_words,
            "validation_scheme": "Stratified 5-Fold Cross-Validation (20 samples/word, 4 samples/fold)",
            "majority_class_baseline": 0.500,
        },
        "per_word_metrics": {},
        "aggregate_metrics": {},
    }

    for rep_k in all_rep_keys:
        results["per_word_metrics"][rep_k] = {}
        total_correct_agg = 0
        total_evaluated_agg = 0
        all_silhouettes = []
        all_dprimes = []
        all_noise_stabilities = []

        for word in target_words:
            X = np.array(features_by_word_and_rep[word][rep_k])
            y = np.array(labels_by_word[word])

            acc, n_corr, n_tot = compute_stratified_5fold_cv(X, y)
            ci_low, ci_high = compute_wilson_ci(n_corr, n_tot, confidence=0.95)
            sil = compute_silhouette(X, y)
            dprime = compute_fisher_dprime(X, y)

            stab_scores = []
            for item_vecs in noise_pert_by_word_and_rep[word][rep_k]:
                if len(item_vecs) >= 2:
                    v0, v1 = item_vecs[0], item_vecs[1]
                    norm0, norm1 = np.linalg.norm(v0), np.linalg.norm(v1)
                    if norm0 > 1e-12 and norm1 > 1e-12:
                        stab_scores.append(float(np.dot(v0, v1) / (norm0 * norm1)))
            noise_stab = float(np.mean(stab_scores)) if stab_scores else 1.0

            total_correct_agg += n_corr
            total_evaluated_agg += n_tot
            all_silhouettes.append(sil)
            all_dprimes.append(dprime)
            all_noise_stabilities.append(noise_stab)

            results["per_word_metrics"][rep_k][word] = {
                "accuracy": round(acc, 3),
                "ci_95": [ci_low, ci_high],
                "silhouette": round(sil, 3),
                "fisher_dprime": round(dprime, 3),
                "noise_stability": round(noise_stab, 3),
            }

        agg_acc = total_correct_agg / total_evaluated_agg if total_evaluated_agg > 0 else 0.0
        agg_ci_low, agg_ci_high = compute_wilson_ci(total_correct_agg, total_evaluated_agg, confidence=0.95)

        results["aggregate_metrics"][rep_k] = {
            "mean_accuracy": round(agg_acc, 3),
            "ci_95": [agg_ci_low, agg_ci_high],
            "total_correct": total_correct_agg,
            "total_samples": total_evaluated_agg,
            "mean_silhouette": round(float(np.mean(all_silhouettes)), 3),
            "mean_fisher_dprime": round(float(np.mean(all_dprimes)), 3),
            "mean_noise_stability": round(float(np.mean(all_noise_stabilities)), 3),
        }

    print_audit_report(results)

    out_file = os.path.join(flowedit_root, output_json)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"✓ Rigorous audit results saved to {out_file}")

    return results


def print_audit_report(results: Dict[str, Any]):
    print("\n" + "=" * 105)
    print(f"{'PHASE 1 RIGOROUS AUDIT: PERFORMANCE, CONFIDENCE INTERVALS & BASELINES':^105}")
    print("=" * 105)
    print(
        f"{'Representation':<26} | {'CV Acc (95% CI)':<20} | {'Lead':<7} | {'Read':<7} | "
        f"{'Bow':<7} | {'Tear':<7} | {'Sil':<7} | {'d-prime':<7} | {'NoiseStab':<9}"
    )
    print("-" * 105)

    name_map = {
        "raw_char_lookup": "1. Static Char Lookup",
        "sentence_char_pool": "2. Sent-Pooled Chars",
        "buggy_text_embed": "3. Buggy Slice text_embed",
        "corrected_text_embed": "4. Correct text_embed",
        "dit_layer_2": "5. DiT Layer 2",
        "dit_layer_10": "6. DiT Layer 10",
        "dit_layer_14": "7. DiT Layer 14",
        "dit_layer_21": "8. DiT Layer 21",
        "minilm_target": "9. MiniLM Target-Word",
        "minilm_sentence": "10. MiniLM [CLS] Sentence",
    }

    for k, display_name in name_map.items():
        agg = results["aggregate_metrics"][k]
        pwm = results["per_word_metrics"][k]
        acc_str = f"{agg['mean_accuracy']*100:.1f}% [{agg['ci_95'][0]*100:.0f}-{agg['ci_95'][1]*100:.0f}%]"
        lead_acc = f"{pwm['lead']['accuracy']*100:.0f}%"
        read_acc = f"{pwm['read']['accuracy']*100:.0f}%"
        bow_acc = f"{pwm['bow']['accuracy']*100:.0f}%"
        tear_acc = f"{pwm['tear']['accuracy']*100:.0f}%"

        print(
            f"{display_name:<26} | {acc_str:<20} | {lead_acc:<7} | {read_acc:<7} | "
            f"{bow_acc:<7} | {tear_acc:<7} | {agg['mean_silhouette']:<7.3f} | "
            f"{agg['mean_fisher_dprime']:<7.3f} | {agg['mean_noise_stability']:<9.3f}"
        )

    print("=" * 105)
    print("Majority-Class Baseline: 50.0% across all 4 homographs.")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="audit_validation_results.json")
    args = parser.parse_args()
    run_rigorous_audit(device=args.device, output_json=args.output)
