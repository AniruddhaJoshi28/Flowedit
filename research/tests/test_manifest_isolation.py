from pathlib import Path

from research.evaluation.manifest import inference_payload, read_manifest


def test_evaluator_labels_never_enter_inference_payload():
    path = Path(__file__).parents[1] / "manifests" / "homograph_today.jsonl"
    rows = list(read_manifest(path))
    assert len(rows) == 8
    for example, label in rows:
        payload = inference_payload(example)
        assert "label" not in payload
        assert "pronunciation_ipa" not in payload
        assert "sense_id" not in payload
        assert label.example_id == example.example_id
