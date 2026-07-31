"""
Automated All-in-One XTTS Fine-Tuning Runner

Combines dataset preparation, model fine-tuning, and model deployment into a single command.

Usage:
    python finetune/auto_finetune.py --raw_dir ./raw_dataset
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list):
    print(f"\n[RUNNING] {' '.join(cmd)}")
    env = os.environ.copy()
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(description="All-in-One XTTS Fine-Tuning & Deployment")
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Path to folder containing your audio files and metadata.csv")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of fine-tuning epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate (default: 5e-6)")
    parser.add_argument("--base_model_dir", type=str, default="./flowedit/Xtts",
                        help="Path to base XTTS model directory (default: ./flowedit/Xtts)")
    args = parser.parse_args()

    finetune_dir = Path(__file__).parent.resolve()

    raw_dir = Path(args.raw_dir).resolve()
    formatted_dir = Path("./formatted_dataset").resolve()
    output_dir = Path("./finetune_output").resolve()
    base_model_dir = Path(args.base_model_dir).resolve()

    if not raw_dir.exists():
        print(f"Error: Raw dataset folder '{raw_dir}' does not exist!")
        sys.exit(1)

    print("=" * 60)
    print("STARTING AUTOMATED XTTS FINE-TUNING & DEPLOYMENT")
    print("=" * 60)
    print(f"  Raw Dataset:    {raw_dir}")
    print(f"  Base Model Dir: {base_model_dir}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Learning Rate:  {args.lr}")
    print("=" * 60)

    # ── Phase 1: Dataset Preparation ──────────────────────────────
    print("\n▶ PHASE 1: Preparing and formatting dataset...")
    prep_script = finetune_dir / "prepare_dataset.py"
    run_command([
        sys.executable, str(prep_script),
        "--input_dir", str(raw_dir),
        "--output_dir", str(formatted_dir)
    ])

    # ── Phase 2: Fine-Tuning ──────────────────────────────────────
    print("\n▶ PHASE 2: Fine-tuning XTTS-v2 model...")
    ft_script = finetune_dir / "finetune_xtts.py"
    run_command([
        sys.executable, str(ft_script),
        "--dataset_dir", str(formatted_dir),
        "--output_path", str(output_dir),
        "--base_model_dir", str(base_model_dir),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr)
    ])

    # ── Phase 3: Model Deployment ─────────────────────────────────
    print("\n▶ PHASE 3: Deploying fine-tuned checkpoint...")
    deploy_script = finetune_dir / "deploy_finetuned.py"
    run_command([
        sys.executable, str(deploy_script),
        "--finetune_dir", str(output_dir),
        "--target_dir", str(base_model_dir)
    ])

    print("\n" + "=" * 60)
    print("ALL STAGES COMPLETED SUCCESSFULLY!")
    print("Next step: Restart your uvicorn server:")
    print("  fuser -k 8001/tcp")
    print("  uvicorn flowedit.api.main:app --host 0.0.0.0 --port 8001 --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()
