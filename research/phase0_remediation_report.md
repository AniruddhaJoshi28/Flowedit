# Phase 0 remediation report

Date: 2026-10-20. Branch: `research/contextual-homograph-control`.

## Implemented repairs

- `flowedit/backbone/checkpoint_identity.py` authenticates F5 safetensors by tensor schema, depth, hidden dimension, and mel dimension before model construction.
- `F5TTSBackbone._resolve_f5_paths()` auto-discovers only authenticated `.safetensors`; generic XTTS `.pth/.pt` files are no longer candidates. Explicit paths receive the same validation.
- The verified cache checkpoint resolves as F5TTS Base, depth 22, hidden 1024, mel 100, 368 tensors, 337,149,162 stored scalars. The GPU CLI additionally requires SHA-256 `4180310F91D592CEE4BC14998CD37C781F779CF105E8CA8744D9BD48CA7046AE`.
- Every loaded backbone parameter is set to `requires_grad=False`; optimization aborts if any parameter changes bitwise.
- `research/manifests/homograph_today.jsonl` contains eight stable cases, occurrence indices, fixed paired seeds, splits, oracle labels, and two respelling candidates per sense.
- `research.evaluation.manifest.inference_payload()` strips IPA, sense, and oracle respellings. Tests prove label isolation.
- Historical simulated outputs remain untouched. New outputs live under `research/outputs/today/` and contain explicit method/provenance/status fields.
- Project-local `tmp_path` allocation in `tests/conftest.py` avoids the managed Windows mode-0700 pytest failure without suppressing assertions.

## Historical metric reconstruction

`python research/reconstruct_historical_metrics.py` reconstructs:

- CPA mean: 29/32 = 90.625% over four eight-item word subsets.
- Stored workflow semantic transfer: 4/5 = 80%.
- Stored workflow opposite-sense intrusion: 5/5 = 100%.
- Summary table claims 60% transfer and 20% false correction. These do not match the stored per-example workflow and remain invalid for acoustic claims.

Exact opposite-sense failures are `lead_opposite_1`, `lead_opposite_2`, `read_opposite_1`, `bow_opposite_1`, and `tear_opposite_1`: in every case the sole taught sense was retrieved without abstention. A one-prototype softmax margin is 1 by construction and supplies no negative evidence.

## Test results on this CPU host

- New research numerical/data tests: 8 passed, 0 failed, 0 skipped.
- `test_hopfield.py`: 12 passed.
- `test_candidate_eval.py`: 3 passed.
- `test_flowedit_paper_rigor.py`: 3 passed.
- `test_homograph_ema.py`: 20 passed.
- `test_api_endpoints.py`: 4 passed.
- `test_api_schema.py`: 3 passed.
- `test_audio_preprocessor.py`: 6 passed.
- Total completed: 59 passed, 0 assertion failures, 0 skips.
- `test_flowedit_integration.py`: four tests collected; file timed out after 240 seconds after printing one progress dot. It is reported as a timeout, not a pass or assertion failure. The file performs heavyweight model setup and requires separate server diagnosis.

## Commands

```powershell
python -m pytest research\tests -q -p no:cacheprovider
python research\reconstruct_historical_metrics.py
python -c "from flowedit.backbone.f5tts_wrapper import F5TTSBackbone; from flowedit.config import BackboneConfig; print(F5TTSBackbone(BackboneConfig(device='cpu'))._resolve_f5_paths())"
```

GPU synthesis was not executed locally because `torch.cuda.is_available()` is false.

