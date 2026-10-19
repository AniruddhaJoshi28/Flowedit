"""Recompute only arithmetic claims supported by existing raw JSON.

This script intentionally labels semantic bookkeeping separately from acoustic evidence.
It never synthesizes results or treats abstention as a successful correction.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    with (ROOT / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    phase2 = load("phase2_feasibility_results.json")
    word_results = phase2["part_a_cpa"]["per_word_results"]
    cpa = [row["cpa_minilm"]["held_out_nn_acc"] for row in word_results.values()]

    phase3 = load("reports/phase3_prototype_results.json")
    workflow = phase3["learning_loop_results"]
    transfers = [r for word in workflow for r in word["transfer_results"]]
    opposites = [r for word in workflow for r in word["specificity_results"]]
    transfer_success = sum(bool(r["transfer_success"]) for r in transfers)
    false_corrections = sum(bool(r["false_correction"]) for r in opposites)

    reported = phase3["comparative_architecture_benchmark"]["semflow_memory"]
    reconstructed = {
        "cpa_mean": sum(cpa) / len(cpa),
        "cpa_n_test_per_word": 8,
        "semantic_transfer": {
            "successes": transfer_success,
            "total": len(transfers),
            "rate": transfer_success / len(transfers),
        },
        "semantic_false_correction": {
            "errors": false_corrections,
            "total": len(opposites),
            "rate": false_corrections / len(opposites),
        },
        "reported_semflow_summary": reported,
        "warnings": [
            "Phase 3 overall accuracy cannot be independently reconstructed from per-item acoustic predictions.",
            "Runner counts abstention via a hard-coded baseline sense.",
            "Phonetic substitution is an oracle assigned the expected sense.",
            "These figures are semantic routing bookkeeping, not validated phoneme accuracy.",
        ],
    }
    print(json.dumps(reconstructed, indent=2))


if __name__ == "__main__":
    main()

