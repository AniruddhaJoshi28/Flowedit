# FlowEdit: Lifelong Pronunciation Adaptation for TTS

> Implementation of [FlowEdit (arXiv:2606.20518)](https://arxiv.org/abs/2606.20518) adapted for **F5-TTS** backbone.

FlowEdit enables **non-destructive pronunciation correction** in frozen TTS models. Instead of retraining or fine-tuning (which causes catastrophic forgetting), FlowEdit optimizes tiny perturbations in text embedding space and stores them in a Modern Hopfield Network memory.

## Key Features

- **92.7% PER reduction** on mispronounced proper nouns
- **Zero forgetting** — general speech quality is mathematically guaranteed identical
- **~15 seconds** per correction (vs 20+ minutes for fine-tuning)
- **Fuzzy morphological matching** — "Linux" correction applies to "Linux's"
- **Speaker-agnostic** — one correction works across all voices
- **Lifelong** — corrections persist across sessions, stable over 200+ edits

## Architecture

```
Correction Loop:
  Reference Audio → [Whisper Alignment] → Token Indices
                                           ↓
  Text → [Text Encoder] → c → [Optimize δ] → δ* → [Hopfield Memory Write]

Inference:
  Text → [Text Encoder] → c → [Hopfield Refiner (gated)] → ĉ → [XTTS-2 Decoder] → Audio
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Learn a Pronunciation Correction

```bash
python scripts/correct.py \
  --text "My friend Siobhan is visiting" \
  --target-word "Siobhan" \
  --ref-audio ./reference/siobhan_correct.wav \
  --memory-path ./corrections.pt
```

### Synthesize with Corrections

```bash
python scripts/synthesize.py \
  --text "Siobhan's presentation was excellent" \
  --speaker-wav ./speaker_reference.wav \
  --memory-path ./corrections.pt \
  --output ./output.wav
```

### Compare Corrected vs Baseline

```bash
python scripts/synthesize.py \
  --text "Siobhan's presentation was excellent" \
  --speaker-wav ./speaker_reference.wav \
  --memory-path ./corrections.pt \
  --compare --output-dir ./comparison/
```

### Python API

```python
from flowedit.config import FlowEditConfig
from flowedit.pipeline import CorrectionLoop, FlowEditInference

# Learn a correction
config = FlowEditConfig()
loop = CorrectionLoop(config)
loop.load_models()

result = loop.correct(
    text="My friend Siobhan is visiting",
    target_word="Siobhan",
    ref_audio_path="./siobhan_correct.wav",
)

loop.save_memory("./corrections.pt")

# Synthesize with corrections
inference = FlowEditInference(config)
inference.load(memory_path="./corrections.pt")

output = inference.synthesize(
    text="Siobhan's presentation was excellent",
    speaker_wav="./speaker.wav",
    output_path="./output.wav",
)
```

## Project Structure

```
flowedit/
├── config.py              # All hyperparameters (from paper)
├── backbone/
│   └── xtts2_wrapper.py   # Frozen XTTS-2 backbone interface
├── alignment/
│   └── whisper_aligner.py # Stage 1: Whisper forced alignment
├── optimizer/
│   └── latent_optimizer.py# Stage 2: Latent input optimization (δ*)
├── memory/
│   └── hopfield_memory.py # Stage 3: Modern Hopfield Network
├── refiner/
│   └── hopfield_refiner.py# Gated retrieval at inference
├── pipeline/
│   ├── correction_loop.py # Full correction orchestrator
│   └── inference.py       # Inference with memory retrieval
└── utils/
    ├── audio.py           # Mel-spectrogram, audio I/O
    └── metrics.py         # PER, MCD, differentiable losses
```

## How It Works

### Stage 1: Detection & Grounding
Whisper Large-v3 performs forced alignment on reference audio to find the exact token positions of the mispronounced word.

### Stage 2: Latent Input Optimization
A perturbation vector δ is optimized (50 Adam steps) so that `Mel(XTTS2(c + δ))` matches the reference audio. Only target token positions are modified; the rest remain zero.

### Stage 3: Hopfield Memory Write
The optimized δ* is stored in a Modern Hopfield Network as a key-value pair. Keys are text embeddings; values are perturbations.

### Inference
The Hopfield Refiner sits between text encoder and decoder. A similarity gate (sigmoid with learned threshold τ) ensures corrections only activate for matching words — providing **mathematically guaranteed zero forgetting**.

## Configuration

All hyperparameters are in `flowedit/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optimization.n_steps` | 50 | Adam optimization steps |
| `optimization.lr_start` | 0.01 | Initial learning rate |
| `optimization.lambda_reg` | 0.001 | L2 regularization weight |
| `memory.max_entries` | 500 | Max stored corrections |
| `memory.dedup_cosine_threshold` | 0.95 | Dedup similarity threshold |
| `memory.gate_threshold_init` | 5.0 | Gate activation threshold τ |

## Running Tests

```bash
python -m pytest tests/ -v
```

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1
- CUDA-capable GPU (recommended: ≥12GB VRAM)
- ~10GB disk space for model downloads

## Citation

```bibtex
@article{singh2026flowedit,
  title={FlowEdit: Associative Memory for Lifelong Pronunciation Adaptation in Flow-Matching TTS},
  author={Singh, Harshit and Singh, Ayush Pratap and Mathur, Nityanand},
  journal={arXiv preprint arXiv:2606.20518},
  year={2026}
}
```
