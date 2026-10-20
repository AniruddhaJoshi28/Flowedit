"""Strict genuine-F5 loading and differentiable conditioning experiment runtime."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import re
from typing import Iterator

import torch
import torch.nn.functional as F

from flowedit.backbone.checkpoint_identity import inspect_f5_safetensors


EXPECTED_SHA256 = "4180310F91D592CEE4BC14998CD37C781F779CF105E8CA8744D9BD48CA7046AE"


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_verified_f5(checkpoint: str, vocab: str, device: str):
    """Load only the authenticated F5TTS_Base checkpoint and freeze every weight."""
    identity = inspect_f5_safetensors(checkpoint, expected_architecture="F5TTS_Base")
    actual_hash = sha256_file(checkpoint)
    if actual_hash != EXPECTED_SHA256:
        raise RuntimeError(f"F5 checkpoint SHA-256 mismatch: {actual_hash}")
    if (identity.depth, identity.hidden_dim, identity.mel_dim) != (22, 1024, 100):
        raise RuntimeError(f"Unexpected F5 architecture: {identity}")
    from f5_tts.api import F5TTS

    tts = F5TTS(model="F5TTS_Base", ckpt_file=str(checkpoint), vocab_file=str(vocab), device=device)
    model = tts.ema_model
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Backbone freeze verification failed")
    return tts, identity, actual_hash


def replace_occurrence(text: str, word: str, occurrence: int, replacement: str) -> str:
    matches = list(re.finditer(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE))
    if occurrence < 0 or occurrence >= len(matches):
        raise ValueError(f"occurrence {occurrence} not found for {word!r} in {text!r}")
    match = matches[occurrence]
    return text[: match.start()] + replacement + text[match.end() :]


@contextmanager
def conditioning_delta_hook(transformer, delta: torch.Tensor, token_mask: torch.Tensor) -> Iterator[None]:
    """Add a trainable delta only at selected character-token embeddings."""
    embedding = transformer.text_embed.text_embed

    def hook(module, inputs, output):
        local_delta = delta.to(device=output.device, dtype=output.dtype)
        local_mask = token_mask.to(device=output.device, dtype=output.dtype)
        while local_mask.ndim < output.ndim:
            local_mask = local_mask.unsqueeze(-1)
        length = min(output.shape[1], local_mask.shape[1])
        changed = output.clone()
        changed[:, :length] = changed[:, :length] + local_mask[:, :length] * local_delta
        return changed

    handle = embedding.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def differentiable_sample(
    model,
    cond_wave: torch.Tensor,
    text: str,
    duration: int,
    steps: int,
    cfg_strength: float,
    seed: int,
    delta: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Reproduce the genuine F5 Euler/CFG path while retaining gradients."""
    from f5_tts.model.utils import list_str_to_idx
    from f5_tts.model.utils import convert_char_to_pinyin

    device = model.device
    cond_wave = cond_wave.to(device)
    cond = model.mel_spec(cond_wave).permute(0, 2, 1).to(next(model.parameters()).dtype)
    ref_frames = cond.shape[1]
    text_value = convert_char_to_pinyin([text])
    text_tokens = list_str_to_idx(text_value, model.vocab_char_map).to(device)
    duration = max(duration, ref_frames + 1, text_tokens.shape[1] + 1)
    cond = F.pad(cond, (0, 0, 0, duration - ref_frames))
    cond_mask = torch.zeros((1, duration, 1), dtype=torch.bool, device=device)
    cond_mask[:, :ref_frames] = True
    step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
    generator = torch.Generator(device=device).manual_seed(seed)
    state = torch.randn((1, duration, model.num_channels), generator=generator, device=device, dtype=cond.dtype)
    times = torch.linspace(0, 1, steps + 1, device=device, dtype=cond.dtype)
    transformer = model.transformer
    transformer.clear_cache()

    context = conditioning_delta_hook(transformer, delta, token_mask) if delta is not None else _null_context()
    with context, torch.enable_grad():
        for index in range(steps):
            time = times[index]
            prediction_pair = transformer(
                x=state, cond=step_cond, text=text_tokens, time=time,
                mask=None, cfg_infer=True, cache=False,
            )
            prediction, null_prediction = torch.chunk(prediction_pair, 2, dim=0)
            velocity = prediction + (prediction - null_prediction) * cfg_strength
            state = state + (times[index + 1] - time) * velocity
    transformer.clear_cache()
    state = torch.where(cond_mask, cond, state)
    return state[:, ref_frames:, :], text_tokens, ref_frames


@contextmanager
def _null_context():
    yield


def target_token_mask(ref_text: str, generated_text: str, word: str, occurrence: int, token_length: int, device) -> torch.Tensor:
    matches = list(re.finditer(rf"\b{re.escape(word)}\b", generated_text, flags=re.IGNORECASE))
    if occurrence >= len(matches):
        raise ValueError("target occurrence unavailable")
    match = matches[occurrence]
    offset = len(ref_text)
    mask = torch.zeros((1, token_length), device=device)
    start = min(token_length, offset + match.start())
    end = min(token_length, offset + match.end())
    mask[:, start:end] = 1
    if not mask.any():
        raise RuntimeError("target token mask is empty; tokenizer alignment must be inspected")
    return mask
