# Phase 0 repository and scientific audit

Status: **gate not yet passed** (2026-10-20). This document records verified evidence; it does not validate the historical performance claims.

## Research question

Does this repository contain a reproducible, genuinely pretrained, frozen F5-TTS path and raw evidence sufficient to support the reported homograph-correction results?

## Repository state and isolation

- Outer repository: `main` at `b33efbb`, dirty before this work (`Flowedit`, `corrections.pt`, and untracked artifacts).
- Inner repository: started dirty on `feature/AJ` at `c2292dc` with one modified wrapper and numerous untracked experiment files.
- New inner branch: `research/contextual-homograph-control`. No historical artifact was overwritten or committed.
- There is no repository `AGENTS.md`.

## Runtime and hardware

| Item | Observed value |
|---|---|
| OS | Windows 11 `10.0.26300` |
| Python | 3.12.10 |
| PyTorch | 2.14.0 |
| torchaudio | 2.11.0 |
| f5-tts | 1.1.22 |
| torchdiffeq | 0.2.5 |
| CUDA | unavailable (`torch.cuda.is_available() == False`) |
| ffmpeg | not found by pydub |

This CPU-only host is adequate for source audit and small forward/JVP checks, but not for the paper-scale 50-step differentiable ODE optimization. Any such experiment must be explicitly reduced or moved to a GPU host.

## Genuine checkpoint and architecture

The official `F5TTS(model="F5TTS_Base")` API resolves:

`%USERPROFILE%/.cache/huggingface/hub/models--SWivid--F5-TTS/snapshots/84e5a410d9cead4de2f847e7c9369a6440bdfaca/F5TTS_Base/model_1200000.safetensors`

- Size: 1,348,645,281 bytes.
- SHA-256: `4180310F91D592CEE4BC14998CD37C781F779CF105E8CA8744D9BD48CA7046AE`.
- Historical probe metadata reports 337,096,804 trainable/model parameters, hidden size 1024, depth 22, vocabulary 2,545, CPU execution. Independent safetensors inspection found 368 tensors and 337,149,162 stored scalars (including non-parameter mel buffers), consistent in scale but not a substitute for a fresh loaded-parameter enumeration.
- The project-local `models/f5tts/.../F5TTS_Base/` contains only `vocab.txt`, not F5 weights.

The generic `models/` folder contains unrelated XTTS weights (`base_model.pth`, 1.064 GB; `model.pth`, 1.031 GB). `F5TTSBackbone._resolve_f5_paths()` recursively searches that generic folder for any large `.pt/.pth/.bin/.safetensors` and can rank one of these unrelated files ahead of the valid HF fallback. This is a correctness defect: the wrapper does not authenticate architecture or state-dict schema before selection.

## Genuine sampling path

Installed `f5_tts.model.cfm.CFM.sample` is decorated with `torch.no_grad()`. It:

1. converts the reference waveform to a 100-channel mel condition;
2. tokenizes text at character level (`-1` padding);
3. pads the mel condition to requested duration;
4. draws the source noise, deterministically when a seed is supplied;
5. constructs conditional and unconditional DiT predictions for classifier-free guidance;
6. integrates with `torchdiffeq.odeint`; the configured method is Euler;
7. uses 32 steps by default and optionally warps time using sway sampling or EPSS;
8. restores the reference frames and decodes with Vocos.

F5-TTS does not expose a phoneme-to-frame alignment. Text characters and acoustic frames are different axes. Any repository code that maps target characters to frames by proportional interpolation is a heuristic, not an alignment furnished by F5-TTS.

The repository's differentiable wrapper bypasses `CFM.sample` and manually reimplements Euler integration. It must be validated against the official solver under identical conditioning, CFG, duration, time grid, caching, and noise. The FlowEdit paper states adjoint differentiation, but the repository path inspected so far uses unrolled autograd/checkpointing; equivalence has not been established.

## Frozen status

Historical scripts call the pretrained model in evaluation or `no_grad` modes in several places, but “frozen” is not established by a saved before/after model hash or parameter checksum. The new experiment must record parameter hashes before and after and assert that every backbone parameter has `requires_grad=False`. Do not infer frozen status merely because no optimizer was constructed over model parameters.

## Historical evidence audit

### Invalid or unsupported artifacts

1. `flowedit/evaluation/run_benchmark.py` explicitly labels its loop simulated and writes fixed values (`WER=0`, `PER=0`, speaker similarity `0.92`, duration `2.5`) without synthesis. Consequently `reports/baseline_results.json` and `reports/final_results.json` are not acoustic benchmark results.
2. `scripts/run_phase2_feasibility.py` uses a random normalized mel-space perturbation and says it “simulates” an additive memory bias. Its `part_c_acoustic` result measures change, not pronunciation correction. Its own drift ratios are 1.15, 1.20, 1.22, and 0.32, all failing localization; nevertheless its prose verdict incorrectly says alpha <= 0.5 is localized with drift < 0.05. The numerical gate correctly says NO-GO.
3. The same Phase 2 memory experiment stores synthetic one-hot “phonetic target vectors”; it tests semantic routing, not acoustic correction memory.
4. `scripts/run_phase3_prototype.py` labels phonetic substitution an oracle and sets `pred_sense = expected_sense`, guaranteeing its reported 100%. The baseline sense is also hard-coded by word rather than measured from generated audio.
5. In Phase 3, abstention falls back to a hard-coded default sense and can be counted as correct in overall accuracy. This violates the required selective-prediction accounting.
6. The Phase 3 CPA is instantiated and immediately put in evaluation mode; the runner does not load a trained adapter checkpoint in the inspected path. Claims attributing its results to a trained CPA require additional provenance.
7. The Wav2Vec2 model is an orthographic English CTC ASR model, not a validated phoneme recognizer. Character fragments plus hand-coded formant rules cannot independently establish phoneme accuracy.
8. The report claims a blinded 15/16 human-listener audit, but no listener-level judgments, blinding protocol, identities/count, or raw annotation file was found in the inspected artifacts.
9. The 177 WAV files are real files, but existence is not evidence that corresponding labels, alignments, or scores are valid.

### Reconstructed headline metrics

The JSON arithmetic itself is reproducible:

- contextual adapter: mean `(0.875 + 0.875 + 1.0 + 0.875)/4 = 0.90625`, reported 90.6%; only 8 test items per word and the split construction still needs leakage audit;
- the Phase 3 summary reports overall 73.3%, transfer 60%, and false correction 20%; however, its own stored per-example workflow reconstructs **4/5 semantic transfers (80%) and 5/5 opposite-sense false corrections (100%)**. The summary and per-example paths are inconsistent and must not be merged into a claim;
- none of these are independent acoustic pronunciation accuracies because of the oracle/default/abstention rules above.

## Immediate defects and leakage risks

- unsafe checkpoint auto-discovery across incompatible model families;
- inconsistent documentation of F5-TTS size (335M, 337M, and 600M all appear);
- character-to-frame proportional mapping presented as localization;
- different initial noise used for the phonetic-reference pass in Phase 2, confounding text and noise;
- a single-prototype Hopfield margin is hard-coded to 1.0, so the margin cannot detect opposite-sense intrusion;
- word-filter fallback uses all entries when no spelling match exists, creating a possible cross-word retrieval path;
- preselected, tiny sentence templates and likely training/evaluation reuse require manifest-level group splitting;
- no frozen final test manifest exists;
- no raw human listening data exists;
- historical paper-ready recommendation is not supported by the raw evidence.

## Phase 0 decision

**Do not advance to a large benchmark.** The genuine checkpoint exists, but the wrapper identity defect, missing independent frozen/parameter verification, unvalidated custom solver, and invalid historical evaluation prevent the Phase 0 gate from passing. The next legitimate step is a small, paired, real-checkpoint actuator test using the official model internals, identical noise and conditioning, parameter hash assertions, and raw mel/audio outputs.

## Reproduction commands

```powershell
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import importlib.metadata as m; print(m.version('f5-tts'))"
Get-FileHash "$env:USERPROFILE\.cache\huggingface\hub\models--SWivid--F5-TTS\snapshots\84e5a410d9cead4de2f847e7c9369a6440bdfaca\F5TTS_Base\model_1200000.safetensors" -Algorithm SHA256
python research/reconstruct_historical_metrics.py
```
