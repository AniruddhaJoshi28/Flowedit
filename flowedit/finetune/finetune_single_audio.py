"""
Single-Audio Fine-Tuning Helper

Takes a single audio clip (with perfect pronunciation) and its text,
automatically sets up the dataset, fine-tunes XTTS for 50 epochs,
and deploys the fine-tuned model into FlowEdit.

Usage:
    python finetune/finetune_single_audio.py \
        --audio_path /path/to/correct_audio.wav \
        --text "My name is Mrunmayee Sakharwade."
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune XTTS on a single reference audio clip")
    parser.add_argument("--audio_path", type=str, required=True,
                        help="Path to the WAV audio file with correct pronunciation")
    parser.add_argument("--text", type=str, required=True,
                        help="Exact text spoken in the audio file")
    parser.add_argument("--speaker_name", type=str, default="target_speaker",
                        help="Speaker label (default: target_speaker)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of fine-tuning epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate (default: 5e-6)")
    args = parser.parse_args()

    audio_src = Path(args.audio_path).resolve()
    if not audio_src.exists():
        print(f"Error: Audio file not found: {audio_src}")
        sys.exit(1)

    raw_dir = Path("./single_audio_dataset").resolve()
    wavs_dir = raw_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    target_wav = wavs_dir / "audio_sample.wav"
    shutil.copy2(audio_src, target_wav)

    meta_csv = raw_dir / "metadata.csv"
    with open(meta_csv, "w", encoding="utf-8") as f:
        f.write(f"wavs/audio_sample.wav|{args.text}|{args.speaker_name}\n")

    print("=" * 60)
    print("SINGLE-AUDIO XTTS FINE-TUNING")
    print("=" * 60)
    print(f"  Reference Audio: {audio_src}")
    print(f"  Text:            \"{args.text}\"")
    print(f"  Dataset Dir:     {raw_dir}")
    print(f"  Epochs:          {args.epochs}")
    print("=" * 60)

    finetune_dir = Path(__file__).parent.resolve()
    auto_script = finetune_dir / "auto_finetune.py"

    cmd = [
        sys.executable, str(auto_script),
        "--raw_dir", str(raw_dir),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr)
    ]

    res = subprocess.run(cmd)
    if res.returncode != 0:
        print("\n[ERROR] Single-audio fine-tuning failed.")
        sys.exit(res.returncode)

    print("\n" + "=" * 60)
    print("SINGLE-AUDIO FINE-TUNING & DEPLOYMENT COMPLETE!")
    print("Your model has now memorized the exact pronunciation from your audio file!")
    print("Restart your uvicorn server:")
    print("  fuser -k 8001/tcp")
    print("  uvicorn flowedit.api.main:app --host 0.0.0.0 --port 8001 --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()
