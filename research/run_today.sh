#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${FLOWEDIT_F5TTS_CKPT:?Set FLOWEDIT_F5TTS_CKPT to the verified model_1200000.safetensors}"
FLOWEDIT_F5TTS_VOCAB="${FLOWEDIT_F5TTS_VOCAB:-$ROOT/models/f5tts/f5tts/F5TTS_Base/vocab.txt}"
DEVICE="${FLOWEDIT_DEVICE:-cuda:0}"
OUTPUT="${FLOWEDIT_TODAY_OUTPUT:-$ROOT/research/outputs/today}"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("STOP: CUDA is unavailable")
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
for index in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(index)
    print(index, torch.cuda.get_device_name(index), "free_GiB", free/2**30, "total_GiB", total/2**30)
PY

nvidia-smi
python -m research.run_homograph_experiment --mode baseline --device "$DEVICE" \
  --checkpoint "$FLOWEDIT_F5TTS_CKPT" --vocab "$FLOWEDIT_F5TTS_VOCAB" \
  --output "$OUTPUT" --acknowledge-reference-rights

python -m research.run_homograph_experiment --mode oracle-respell --device "$DEVICE" \
  --checkpoint "$FLOWEDIT_F5TTS_CKPT" --vocab "$FLOWEDIT_F5TTS_VOCAB" \
  --output "$OUTPUT" --acknowledge-reference-rights

python -m research.run_homograph_experiment --mode smoke --device "$DEVICE" \
  --checkpoint "$FLOWEDIT_F5TTS_CKPT" --vocab "$FLOWEDIT_F5TTS_VOCAB" \
  --output "$OUTPUT" --acknowledge-reference-rights --sample-id lead_metal_001

# set -e makes the smoke test a hard gate for optimization.
python -m research.run_homograph_experiment --mode optimize --device "$DEVICE" \
  --checkpoint "$FLOWEDIT_F5TTS_CKPT" --vocab "$FLOWEDIT_F5TTS_VOCAB" \
  --output "$OUTPUT" --acknowledge-reference-rights --sample-id lead_metal_001 --optim-steps 10

python -m research.summarize_today --output-root "$OUTPUT" \
  --report "$ROOT/research/today_gpu_experiment_report.md"
