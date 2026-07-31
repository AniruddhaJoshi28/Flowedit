"""
Standalone XTTS diagnostic script.

Tests the fine-tuned XTTS model DIRECTLY via Coqui TTS,
completely bypassing FlowEdit's wrapper, hooks, and correction pipeline.

Usage:
    python test_xtts_direct.py \
        --model_dir ./flowedit/Xtts \
        --speaker_wav /path/to/speaker_reference.wav \
        --text "My name is Mrunmayee Sakharvade." \
        --output_dir ./diagnostic_outputs

If this script produces PERFECT pronunciation, then the issue is
in FlowEdit's wrapper code.  If it also mispronounces, then the
issue is in the model weights, vocab, or reference audio.
"""

import argparse
import os
import sys
import torch
import time


def main():
    parser = argparse.ArgumentParser(description="Direct XTTS inference (no FlowEdit)")
    parser.add_argument("--model_dir", type=str, default="./flowedit/Xtts",
                        help="Path to XTTS model directory (contains model.pth, config.json, vocab.json)")
    parser.add_argument("--speaker_wav", type=str, required=True,
                        help="Path to speaker reference audio WAV file")
    parser.add_argument("--text", type=str, default="My name is Mrunmayee Sakharvade.",
                        help="Text to synthesize")
    parser.add_argument("--language", type=str, default="en",
                        help="Language code")
    parser.add_argument("--output_dir", type=str, default="./diagnostic_outputs",
                        help="Directory to save output WAV files")
    # Inference parameters — try multiple combinations
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature (default: try multiple)")
    parser.add_argument("--repetition_penalty", type=float, default=None,
                        help="Override repetition_penalty (default: try multiple)")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--gpt_cond_len", type=int, default=None,
                        help="GPT conditioning length (default: try multiple)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────
    print("=" * 60)
    print("XTTS DIRECT INFERENCE DIAGNOSTIC")
    print("=" * 60)
    print(f"  Model dir:    {args.model_dir}")
    print(f"  Speaker WAV:  {args.speaker_wav}")
    print(f"  Text:         {args.text}")
    print(f"  Language:     {args.language}")
    print()

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config_path = os.path.join(args.model_dir, "config.json")
    checkpoint_path = os.path.join(args.model_dir, "model.pth")
    vocab_path = os.path.join(args.model_dir, "vocab.json")

    for f in [config_path, checkpoint_path, vocab_path]:
        if not os.path.isfile(f):
            print(f"ERROR: Missing file: {f}")
            sys.exit(1)

    print("Loading XTTS model...")
    xtts_config = XttsConfig()
    xtts_config.load_json(config_path)

    model = Xtts.init_from_config(xtts_config)
    model.load_checkpoint(
        xtts_config,
        checkpoint_dir=args.model_dir,
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        eval=True,
        use_deepspeed=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}")

    # ── Extract speaker conditioning ────────────────────────────────
    print(f"\nExtracting speaker conditioning from: {args.speaker_wav}")

    # Define parameter grids to test
    gpt_cond_lens = [args.gpt_cond_len] if args.gpt_cond_len else [6, 12, 24, 30]
    temperatures = [args.temperature] if args.temperature else [0.1, 0.3, 0.5, 0.65, 0.75, 0.85]
    rep_penalties = [args.repetition_penalty] if args.repetition_penalty else [2.0, 5.0, 10.0]

    results = []

    for gpt_cond_len in gpt_cond_lens:
        print(f"\n--- gpt_cond_len={gpt_cond_len} ---")
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[args.speaker_wav],
            gpt_cond_len=gpt_cond_len,
            gpt_cond_chunk_len=4,
            max_ref_length=30,
        )
        print(f"  gpt_cond_latent shape: {gpt_cond_latent.shape}")
        print(f"  speaker_embedding shape: {speaker_embedding.shape}")

        for temp in temperatures:
            for rep_pen in rep_penalties:
                label = f"cond{gpt_cond_len}_temp{temp}_rep{rep_pen}"
                output_path = os.path.join(args.output_dir, f"{label}.wav")

                print(f"\n  Generating: temp={temp}, rep_penalty={rep_pen}, "
                      f"top_k={args.top_k}, top_p={args.top_p}")

                start = time.time()
                out = model.inference(
                    text=args.text,
                    language=args.language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=temp,
                    length_penalty=1.0,
                    repetition_penalty=rep_pen,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    enable_text_splitting=False,
                )
                elapsed = time.time() - start

                wav = out["wav"]
                if isinstance(wav, torch.Tensor):
                    wav_np = wav.squeeze().cpu().numpy()
                else:
                    import numpy as np
                    wav_np = np.array(wav)

                import soundfile as sf
                sr = 24000  # XTTS output sample rate
                sf.write(output_path, wav_np, sr)

                duration = len(wav_np) / sr
                print(f"    -> Saved: {output_path} ({duration:.2f}s, took {elapsed:.1f}s)")

                results.append({
                    "file": output_path,
                    "gpt_cond_len": gpt_cond_len,
                    "temperature": temp,
                    "repetition_penalty": rep_pen,
                    "duration": duration,
                    "inference_time": elapsed,
                })

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"DIAGNOSTIC COMPLETE: {len(results)} audio files generated")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    print("\nListen to each file and identify which parameter combination")
    print("produces the CORRECT pronunciation. Then we'll match those")
    print("exact parameters in FlowEdit's xtts_wrapper.py.\n")

    for r in results:
        print(f"  {os.path.basename(r['file']):40s}  "
              f"cond_len={r['gpt_cond_len']:2d}  "
              f"temp={r['temperature']:.2f}  "
              f"rep={r['repetition_penalty']:.1f}  "
              f"dur={r['duration']:.2f}s")


if __name__ == "__main__":
    main()
