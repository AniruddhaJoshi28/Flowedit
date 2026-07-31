"""
XTTS Dataset Preparation Helper

Preprocesses audio files and builds the metadata CSV files for XTTS fine-tuning.

Input format:
    A folder containing:
    1. Audio files (.wav, .mp3, .flac, etc.)
    2. A transcriptions file (e.g. metadata.txt or transcriptions.csv) formatted as:
       filename|text
       OR
       filename|text|speaker_name

Usage:
    python prepare_dataset.py \
        --input_dir ./raw_dataset \
        --output_dir ./formatted_dataset \
        --val_ratio 0.1
"""

import argparse
import os
import random
import sys
from pathlib import Path
import librosa
import soundfile as sf


def preprocess_audio(input_path: Path, output_path: Path, target_sr: int = 22050):
    """Load audio, convert to mono, resample to target_sr, save as PCM_16 WAV."""
    y, sr = librosa.load(input_path, sr=target_sr, mono=True)
    # Trim silence at start/end
    y, _ = librosa.effects.trim(y, top_db=30)
    sf.write(output_path, y, target_sr, subtype="PCM_16")
    duration = len(y) / target_sr
    return duration


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for XTTS fine-tuning")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to folder containing raw audio and metadata file")
    parser.add_argument("--output_dir", type=str, default="./formatted_dataset",
                        help="Path to store processed WAVs and tts_train.csv / tts_val.csv")
    parser.add_argument("--metadata_file", type=str, default=None,
                        help="Name of metadata file inside input_dir (default: metadata.csv or metadata.txt)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of data to reserve for validation (default: 0.1)")
    parser.add_argument("--speaker_name", type=str, default="custom_speaker",
                        help="Default speaker name if not specified in metadata")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    wavs_out_dir = output_dir / "wavs"
    wavs_out_dir.mkdir(parents=True, exist_ok=True)

    # Locate metadata file
    meta_path = None
    if args.metadata_file:
        meta_path = input_dir / args.metadata_file
    else:
        for possible_name in ["metadata.csv", "metadata.txt", "transcriptions.txt", "transcriptions.csv"]:
            if (input_dir / possible_name).exists():
                meta_path = input_dir / possible_name
                break

    if not meta_path or not meta_path.exists():
        print(f"Error: Could not find metadata file in {input_dir}.")
        print("Please provide a metadata file formatted as 'filename|text' or 'filename|text|speaker'.")
        sys.exit(1)

    print(f"Reading metadata from: {meta_path}")

    items = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                file_name = parts[0].strip()
                text = parts[1].strip()
                speaker = parts[2].strip() if len(parts) >= 3 else args.speaker_name
                items.append((file_name, text, speaker))
            else:
                print(f"Warning: Line {line_num} does not have at least 2 pipe-separated fields. Skipping: '{line}'")

    if not items:
        print("Error: No valid entries found in metadata file.")
        sys.exit(1)

    print(f"Found {len(items)} audio entries. Preprocessing audio to 22,050Hz Mono WAV...")

    valid_entries = []
    for file_name, text, speaker in items:
        # Find actual file (check inside input_dir or input_dir/wavs)
        audio_src = None
        for p in [input_dir / file_name, input_dir / "wavs" / file_name]:
            if p.exists():
                audio_src = p
                break
            # Try adding .wav extension if missing
            if not file_name.endswith(".wav"):
                p_wav = Path(str(p) + ".wav")
                if p_wav.exists():
                    audio_src = p_wav
                    file_name = file_name + ".wav"
                    break

        if not audio_src:
            print(f"  [MISSING] Audio file not found: {file_name}. Skipping.")
            continue

        clean_filename = Path(file_name).name
        target_wav_path = wavs_out_dir / clean_filename

        try:
            duration = preprocess_audio(audio_src, target_wav_path)
            if duration < 0.5:
                print(f"  [SHORT] {clean_filename} is too short ({duration:.2f}s). Skipping.")
                continue
            rel_wav_path = f"wavs/{clean_filename}"
            valid_entries.append(f"{rel_wav_path}|{text}|{speaker}")
            print(f"  [OK] {clean_filename} ({duration:.2f}s) -> '{text}'")
        except Exception as e:
            print(f"  [ERROR] Processing {clean_filename} failed: {e}")

    if not valid_entries:
        print("Error: No audio files were successfully processed.")
        sys.exit(1)

    # Shuffle and split train / val
    random.seed(42)
    random.shuffle(valid_entries)

    n_val = max(1, int(len(valid_entries) * args.val_ratio))
    val_entries = valid_entries[:n_val]
    train_entries = valid_entries[n_val:] if len(valid_entries) > 1 else valid_entries

    train_csv = output_dir / "tts_train.csv"
    val_csv = output_dir / "tts_val.csv"

    header = "audio_file|text|speaker_name\n"
    with open(train_csv, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(train_entries) + "\n")

    with open(val_csv, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(val_entries) + "\n")

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Total processed: {len(valid_entries)}")
    print(f"Train samples:   {len(train_entries)} ({train_csv})")
    print(f"Validation:      {len(val_entries)} ({val_csv})")
    print(f"Output directory:{output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
