# Same-day controllability experiment

## Executable stages

1. `baseline`: eight genuine F5 WAVs with original spelling; inference receives no pronunciation label.
2. `oracle-respell`: sixteen WAVs, two declared candidates per target pronunciation. These are oracle actuator controls, never end-to-end predictions.
3. `smoke`: measured Whisper word boundary, paired original/respelling counterfactual mels with identical reference, seed, duration, CFG, and solver budget; one trainable 512-dimensional target-character conditioning delta; nonzero finite gradient and bitwise-frozen backbone required.
4. `optimize`: 10, 25, or 50 bounded experiment steps after smoke success. Saves every loss component, gradient norm, final delta, WAV, peak memory, and checkpoint identity.

The differentiable sampler reproduces F5's actual mel frontend, character tokenizer, DiT, conditional/unconditional CFG, and Euler updates. It intentionally disables transformer caching during gradient computation. It does not call the official `@torch.no_grad()` sampler.

## Loss and scope

For target mask `T`, protected mask `P`, paired oracle-respelling mel `y*`, baseline mel `y0`, and generated mel `y(delta)`:

`L = MSE_T(y(delta), y*) + lambda MSE_P(y(delta), y0) + mu ||delta||²`.

Whisper timestamps define the word interval. A 40 ms uncertainty band is removed from both the inner target and protected regions. This is a baseline objective, not a novelty claim. Reduced loss does not prove a correct phoneme; the saved WAV still requires validated phonetic or blinded listening judgment.

## IIT execution

```bash
python -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
# Install the CUDA torch/torchaudio wheels compatible with nvidia-smi first.
python -m pip install -r research/requirements-gpu.txt
export FLOWEDIT_F5TTS_CKPT=/absolute/path/F5TTS_Base/model_1200000.safetensors
export FLOWEDIT_F5TTS_VOCAB="$PWD/models/f5tts/f5tts/F5TTS_Base/vocab.txt"
export FLOWEDIT_DEVICE=cuda:0
bash research/run_today.sh
```

The operator must verify the license/consent status of `scratch/ref_trimmed.wav` before passing the CLI acknowledgement. The code cannot establish legal provenance from repository contents.

## Stop conditions enforced

- CUDA unavailable; checkpoint schema/hash mismatch; missing baseline; failed word alignment; absent/nonfinite/zero delta gradient; changed backbone parameter; missing reference rights acknowledgement; or a failed shell stage.

## Evaluation required after execution

Listen blind to baseline and all declared respelling variants on the development cases before selecting a variant. Freeze that selection before test evaluation. Record listener-level votes. Report oracle actuator feasibility separately from autonomous contextual selection.

