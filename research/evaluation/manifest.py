"""Separate inference inputs from evaluator-only pronunciation labels."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class InferenceExample:
    example_id: str
    text: str
    target_word: str
    reference_audio: str
    reference_text: str
    seed: int
    occurrence_index: int
    split: str


@dataclass(frozen=True)
class EvaluationLabel:
    example_id: str
    pronunciation_ipa: str
    sense_id: str
    notes: str
    oracle_respellings: tuple[str, ...]
    reference_rights: str


def read_manifest(path: str | Path) -> Iterator[tuple[InferenceExample, EvaluationLabel]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"id", "text", "target_word", "occurrence_index", "reference_audio", "reference_text", "seed", "split", "label"}
            missing = required - row.keys()
            if missing:
                raise ValueError(f"line {line_number} missing fields: {sorted(missing)}")
            label = row["label"]
            yield (
                InferenceExample(
                    example_id=row["id"], text=row["text"], target_word=row["target_word"],
                    reference_audio=row["reference_audio"], reference_text=row["reference_text"],
                    seed=int(row["seed"]),
                    occurrence_index=int(row["occurrence_index"]), split=row["split"],
                ),
                EvaluationLabel(
                    example_id=row["id"], pronunciation_ipa=label["pronunciation_ipa"],
                    sense_id=label["sense_id"], notes=label.get("notes", ""),
                    oracle_respellings=tuple(label.get("oracle_respellings", ())),
                    reference_rights=label.get("reference_rights", "unknown"),
                ),
            )


def inference_payload(example: InferenceExample) -> dict:
    """Return the only fields permitted to cross into autonomous inference."""
    return {
        "id": example.example_id,
        "text": example.text,
        "target_word": example.target_word,
        "reference_audio": example.reference_audio,
        "reference_text": example.reference_text,
        "seed": example.seed,
        "occurrence_index": example.occurrence_index,
        "split": example.split,
    }
