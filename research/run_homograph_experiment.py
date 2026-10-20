"""Same-day genuine F5-TTS homograph actuator experiment.

Ground-truth pronunciation labels are used only in explicitly ORACLE respelling
and reference-matching modes. Baseline inference never receives them.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from research.evaluation.manifest import read_manifest
from research.f5_runtime import (
    differentiable_sample,
    load_verified_f5,
    replace_occurrence,
    target_token_mask,
)


ROOT = Path(__file__).resolve().parents[1]


def json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def gpu_stats(device: str) -> dict:
    if not device.startswith("cuda"):
        return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def load_reference(path: Path, device: str, target_rate: int = 24000) -> torch.Tensor:
    audio, rate = torchaudio.load(str(path))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    rms = audio.square().mean().sqrt()
    if rms < 0.1:
        audio = audio * (0.1 / rms.clamp_min(1e-8))
    if rate != target_rate:
        audio = torchaudio.functional.resample(audio, rate, target_rate)
    return audio.to(device)


def expected_duration_frames(reference: torch.Tensor, reference_text: str, generated_text: str, speed: float = 1.0) -> int:
    reference_frames = reference.shape[-1] // 256
    return reference_frames + int(
        reference_frames / max(1, len(reference_text.encode("utf-8")))
        * len(generated_text.encode("utf-8")) / speed
    )


def run_generation(args, tts, identity, checkpoint_hash, mode: str) -> None:
    manifest = list(read_manifest(args.manifest))
    output_root = args.output / mode
    output_root.mkdir(parents=True, exist_ok=True)
    for example, label in manifest:
        variants = [("original", example.text)]
        if mode == "oracle_respell":
            variants = [
                (f"variant_{index + 1}_{spelling}", replace_occurrence(
                    example.text, example.target_word, example.occurrence_index, spelling
                ))
                for index, spelling in enumerate(label.oracle_respellings)
            ]
        for variant_name, synthesis_text in variants:
            stem = f"{example.example_id}__{variant_name}"
            wav_path = output_root / f"{stem}.wav"
            metadata_path = output_root / f"{stem}.json"
            if wav_path.exists() and metadata_path.exists() and not args.force:
                print(f"RESUME {wav_path}")
                continue
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(args.device)
                torch.cuda.synchronize(args.device)
            started = time.perf_counter()
            wav, sample_rate, mel = tts.infer(
                ref_file=str((ROOT / example.reference_audio).resolve()),
                ref_text=example.reference_text,
                gen_text=synthesis_text,
                nfe_step=args.nfe_steps,
                cfg_strength=args.cfg_strength,
                sway_sampling_coef=args.sway,
                speed=1.0,
                seed=example.seed,
            )
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            elapsed = time.perf_counter() - started
            sf.write(wav_path, wav, sample_rate)
            record = {
                "status": "completed",
                "method": "genuine_f5tts_baseline" if mode == "baseline" else "ORACLE_text_respelling",
                "sample_id": example.example_id,
                "input_text": synthesis_text,
                "original_text": example.text,
                "target_word": example.target_word,
                "oracle_label_used": mode != "baseline",
                "variant": variant_name,
                "seed": example.seed,
                "reference_audio": example.reference_audio,
                "reference_text": example.reference_text,
                "checkpoint": asdict(identity),
                "checkpoint_sha256": checkpoint_hash,
                "nfe_steps": args.nfe_steps,
                "cfg_strength": args.cfg_strength,
                "sway": args.sway,
                "runtime_seconds": elapsed,
                "gpu": gpu_stats(args.device),
                "wav": str(wav_path),
                "mel_shape": list(np.asarray(mel).shape),
                "pronunciation_assessment": "pending_blinded_or_validated_phonetic_evaluation",
            }
            json_write(metadata_path, record)
            print(f"WROTE {wav_path} ({elapsed:.2f}s)")


def locate_word_with_whisper(wav_path: Path, target_word: str, occurrence: int, model_size: str) -> dict:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is required for measured word boundaries") from exc
    aligner = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = aligner.transcribe(str(wav_path), language="en", word_timestamps=True)
    matches = []
    for segment in segments:
        for word in segment.words or []:
            normalized = "".join(character for character in word.word.lower() if character.isalpha())
            if normalized == target_word.lower():
                matches.append((float(word.start), float(word.end), float(word.probability or 0.0)))
    if occurrence >= len(matches):
        raise RuntimeError(f"Whisper did not align occurrence {occurrence} of {target_word!r} in {wav_path}")
    start, end, probability = matches[occurrence]
    return {"start_seconds": start, "end_seconds": end, "word_probability": probability, "uncertainty_seconds": 0.04}


def alignment_masks(alignment: dict, frames: int, duration_seconds: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    scale = frames / max(duration_seconds, 1e-8)
    uncertainty = alignment["uncertainty_seconds"]
    inner_start = max(0, int((alignment["start_seconds"] + uncertainty) * scale))
    inner_end = min(frames, max(inner_start + 1, int((alignment["end_seconds"] - uncertainty) * scale)))
    outer_start = max(0, int((alignment["start_seconds"] - uncertainty) * scale))
    outer_end = min(frames, max(outer_start + 1, int((alignment["end_seconds"] + uncertainty) * scale)))
    target = torch.zeros((1, frames, 1), dtype=torch.bool, device=device)
    protected = torch.ones((1, frames, 1), dtype=torch.bool, device=device)
    target[:, inner_start:inner_end] = True
    protected[:, outer_start:outer_end] = False
    return target, protected


def run_gradient_or_optimization(args, tts, identity, checkpoint_hash, optimize: bool) -> None:
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Gradient experiment requires an available CUDA device; synthetic fallback is forbidden")
    rows = {example.example_id: (example, label) for example, label in read_manifest(args.manifest)}
    example, label = rows[args.sample_id]
    if not label.oracle_respellings:
        raise RuntimeError("No oracle acoustic target is defined")
    baseline_wav = args.output / "baseline" / f"{example.example_id}__original.wav"
    if not baseline_wav.exists():
        raise RuntimeError(f"Run baseline first: missing {baseline_wav}")
    reference = load_reference((ROOT / example.reference_audio).resolve(), args.device)
    combined_original = example.reference_text + (" " if not example.reference_text.endswith(" ") else "") + example.text
    oracle_text = replace_occurrence(example.text, example.target_word, example.occurrence_index, label.oracle_respellings[0])
    combined_oracle = example.reference_text + (" " if not example.reference_text.endswith(" ") else "") + oracle_text
    duration = expected_duration_frames(reference, example.reference_text, example.text)
    model = tts.ema_model
    with torch.no_grad():
        baseline_mel, tokens, _ = differentiable_sample(
            model, reference, combined_original, duration, args.optim_nfe_steps,
            args.cfg_strength, example.seed,
        )
        oracle_mel, _, _ = differentiable_sample(
            model, reference, combined_oracle, duration, args.optim_nfe_steps,
            args.cfg_strength, example.seed,
        )
    token_mask = target_token_mask(
        example.reference_text + (" " if not example.reference_text.endswith(" ") else ""),
        example.text, example.target_word, example.occurrence_index, tokens.shape[1], args.device,
    )
    alignment = locate_word_with_whisper(baseline_wav, example.target_word, example.occurrence_index, args.whisper_model)
    baseline_audio, baseline_rate = sf.read(baseline_wav)
    target_mask, protected_mask = alignment_masks(
        alignment, baseline_mel.shape[1], len(baseline_audio) / baseline_rate, args.device
    )
    delta = torch.zeros((1, 1, 512), device=args.device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=args.learning_rate)
    iterations = args.optim_steps if optimize else 1
    history = []
    initial_weights = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        prediction, _, _ = differentiable_sample(
            model, reference, combined_original, duration, args.optim_nfe_steps,
            args.cfg_strength, example.seed, delta, token_mask,
        )
        common = min(prediction.shape[1], oracle_mel.shape[1], baseline_mel.shape[1])
        prediction = prediction[:, :common]
        target = target_mask[:, :common].expand_as(prediction)
        protected = protected_mask[:, :common].expand_as(prediction)
        pronunciation_loss = F.mse_loss(prediction[target], oracle_mel[:, :common][target])
        preservation_loss = F.mse_loss(prediction[protected], baseline_mel[:, :common][protected])
        regularization = delta.square().mean()
        loss = pronunciation_loss + args.preservation_weight * preservation_loss + args.regularization_weight * regularization
        loss.backward()
        gradient_norm = float(delta.grad.norm().detach().cpu()) if delta.grad is not None else 0.0
        finite = delta.grad is not None and bool(torch.isfinite(delta.grad).all())
        history.append({
            "iteration": iteration, "loss": float(loss.detach().cpu()),
            "oracle_reference_mse": float(pronunciation_loss.detach().cpu()),
            "protected_mse": float(preservation_loss.detach().cpu()),
            "regularization": float(regularization.detach().cpu()),
            "gradient_norm": gradient_norm, "gradient_finite": finite,
        })
        if not finite or gradient_norm == 0.0:
            raise RuntimeError(f"STOP: invalid conditioning gradient at iteration {iteration}: {history[-1]}")
        optimizer.step()

    changed_weights = [name for name, parameter in model.named_parameters() if not torch.equal(parameter, initial_weights[name])]
    if changed_weights:
        raise RuntimeError(f"STOP: frozen backbone weights changed: {changed_weights[:3]}")
    with torch.no_grad():
        final_mel, _, _ = differentiable_sample(
            model, reference, combined_original, duration, args.optim_nfe_steps,
            args.cfg_strength, example.seed, delta, token_mask,
        )
        generated_wave = tts.vocoder.decode(final_mel.permute(0, 2, 1)).squeeze().float().cpu().numpy()
    method = "localized_optimization" if optimize else "gradient_smoke"
    output_dir = args.output / method
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"{example.example_id}.wav"
    sf.write(wav_path, generated_wave, 24000)
    json_write(output_dir / f"{example.example_id}.json", {
        "status": "completed",
        "method": method,
        "oracle_label_used": True,
        "claim_scope": "acoustic_reference_matching_only_not_validated_pronunciation",
        "sample_id": example.example_id,
        "oracle_respelling": label.oracle_respellings[0],
        "alignment": alignment,
        "checkpoint": asdict(identity),
        "checkpoint_sha256": checkpoint_hash,
        "history": history,
        "delta_norm": float(delta.detach().norm().cpu()),
        "frozen_weights_changed": changed_weights,
        "wav": str(wav_path),
        "gpu": gpu_stats(args.device),
    })
    print(f"WROTE {wav_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "oracle-respell", "smoke", "optimize"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "research/manifests/homograph_today.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "research/outputs/today")
    parser.add_argument("--acknowledge-reference-rights", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--nfe-steps", type=int, default=32)
    parser.add_argument("--optim-nfe-steps", type=int, default=8)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway", type=float, default=-1.0)
    parser.add_argument("--sample-id", default="lead_metal_001")
    parser.add_argument("--optim-steps", type=int, choices=(10, 25, 50), default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--preservation-weight", type=float, default=1.0)
    parser.add_argument("--regularization-weight", type=float, default=1e-3)
    parser.add_argument("--whisper-model", default="small.en")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.acknowledge_reference_rights:
        raise SystemExit("Refusing synthesis: pass --acknowledge-reference-rights after verifying speaker consent/license")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; no synthetic fallback will be used")
    tts, identity, checkpoint_hash = load_verified_f5(str(args.checkpoint), str(args.vocab), args.device)
    if args.mode == "baseline":
        run_generation(args, tts, identity, checkpoint_hash, "baseline")
    elif args.mode == "oracle-respell":
        run_generation(args, tts, identity, checkpoint_hash, "oracle_respell")
    elif args.mode == "smoke":
        run_gradient_or_optimization(args, tts, identity, checkpoint_hash, optimize=False)
    else:
        run_gradient_or_optimization(args, tts, identity, checkpoint_hash, optimize=True)


if __name__ == "__main__":
    main()
