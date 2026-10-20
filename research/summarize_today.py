"""Generate an evidence-only report from completed stage metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in sorted(args.output_root.rglob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            records.append((path, {"status": "invalid_json", "error": str(exc)}))
        else:
            records.append((path, record))
    methods: dict[str, list[tuple[Path, dict]]] = {}
    for path, record in records:
        methods.setdefault(record.get("method", "unknown"), []).append((path, record))
    lines = [
        "# Today GPU experiment report",
        "",
        "This report is generated only from saved run metadata. Missing stages are reported as unexecuted.",
        "",
        "## Stage status",
        "",
        "| Stage | Completed records | WAV files | Assessment |",
        "|---|---:|---:|---|",
    ]
    expected = [
        ("genuine_f5tts_baseline", 8),
        ("ORACLE_text_respelling", 16),
        ("gradient_smoke", 1),
        ("localized_optimization", 1),
    ]
    for method, target in expected:
        rows = methods.get(method, [])
        wavs = sum(Path(record.get("wav", "missing")).exists() for _, record in rows)
        assessment = "complete" if len(rows) >= target and wavs >= target else "incomplete/unexecuted"
        lines.append(f"| `{method}` | {len(rows)}/{target} | {wavs}/{target} | {assessment} |")
    lines += ["", "## Recorded measurements", ""]
    if not records:
        lines.append("No GPU records exist. No WAV generation, gradient smoke test, or optimization is claimed.")
    for path, record in records:
        lines += [
            f"### {record.get('method', 'unknown')} - {record.get('sample_id', path.stem)}",
            "",
            f"- Metadata: `{path}`",
            f"- Status: `{record.get('status', 'unknown')}`",
            f"- WAV: `{record.get('wav', 'none')}`",
            f"- Runtime seconds: `{record.get('runtime_seconds', 'not recorded')}`",
            f"- Peak GPU allocated bytes: `{record.get('gpu', {}).get('peak_allocated_bytes', 'not recorded')}`",
            f"- Pronunciation assessment: `{record.get('pronunciation_assessment', record.get('claim_scope', 'not performed'))}`",
            "",
        ]
    lines += [
        "## Scientific decision",
        "",
        "Outcome B unless validated listener/phonetic judgments are attached: implementation is ready, but mel-reference loss alone does not establish pronunciation correction.",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
