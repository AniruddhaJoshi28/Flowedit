"""Fail-closed identity checks for F5-TTS checkpoints.

Auto-discovery must never infer architecture from file size.  This module inspects
the state dictionary without constructing a model and requires F5-specific tensor
names and shapes.  It deliberately rejects opaque formats during auto-discovery.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class F5CheckpointIdentity:
    path: str
    architecture: str
    tensor_count: int
    scalar_count: int
    hidden_dim: int
    mel_dim: int
    depth: int


class CheckpointIdentityError(ValueError):
    """Raised when a candidate cannot be proven to be the requested F5 model."""


def _product(shape: Sequence[int]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


def _infer_depth(keys: Sequence[str]) -> int:
    indices: set[int] = set()
    marker = "transformer.transformer_blocks."
    for key in keys:
        if marker not in key:
            continue
        suffix = key.split(marker, 1)[1]
        head = suffix.split(".", 1)[0]
        if head.isdigit():
            indices.add(int(head))
    if not indices:
        # f5-tts releases have also used a shorter `transformer.blocks` path.
        marker = "transformer.blocks."
        for key in keys:
            if marker in key:
                head = key.split(marker, 1)[1].split(".", 1)[0]
                if head.isdigit():
                    indices.add(int(head))
    return max(indices) + 1 if indices else 0


def inspect_f5_safetensors(path: str | Path, expected_architecture: str | None = None) -> F5CheckpointIdentity:
    """Inspect a safetensors checkpoint and prove it has an F5-TTS DiT schema."""
    from safetensors import safe_open

    candidate = Path(path).expanduser().resolve()
    if candidate.suffix.lower() != ".safetensors":
        raise CheckpointIdentityError(
            f"Automatic F5 checkpoint selection accepts only .safetensors, got: {candidate.name}"
        )
    if not candidate.is_file():
        raise CheckpointIdentityError(f"Checkpoint does not exist: {candidate}")

    with safe_open(str(candidate), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        shapes: Mapping[str, list[int]] = {key: list(handle.get_slice(key).get_shape()) for key in keys}

    def find(suffix: str) -> tuple[str, list[int]]:
        matches = [(key, shape) for key, shape in shapes.items() if key.endswith(suffix)]
        if len(matches) != 1:
            raise CheckpointIdentityError(
                f"Expected one F5 tensor ending in {suffix!r}; found {len(matches)} in {candidate.name}"
            )
        return matches[0]

    _, input_projection = find("transformer.input_embed.proj.weight")
    _, output_projection = find("transformer.proj_out.weight")
    _, text_embedding = find("transformer.text_embed.text_embed.weight")

    if len(input_projection) != 2 or len(output_projection) != 2 or len(text_embedding) != 2:
        raise CheckpointIdentityError("F5 identity tensors must be matrices")
    hidden_dim = int(input_projection[0])
    mel_dim = int(output_projection[0])
    depth = _infer_depth(keys)
    if hidden_dim not in (768, 1024) or mel_dim != 100 or depth < 18:
        raise CheckpointIdentityError(
            f"Incompatible F5 architecture: hidden={hidden_dim}, mel={mel_dim}, depth={depth}"
        )

    architecture = "F5TTS_v1_Base" if "v1" in candidate.as_posix().lower() else "F5TTS_Base"
    if expected_architecture and architecture != expected_architecture:
        raise CheckpointIdentityError(
            f"Checkpoint identifies as {architecture}, requested {expected_architecture}"
        )
    return F5CheckpointIdentity(
        path=str(candidate),
        architecture=architecture,
        tensor_count=len(keys),
        scalar_count=sum(_product(shape) for shape in shapes.values()),
        hidden_dim=hidden_dim,
        mel_dim=mel_dim,
        depth=depth,
    )

