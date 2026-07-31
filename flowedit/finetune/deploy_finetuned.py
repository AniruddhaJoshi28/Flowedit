"""
Deployment Script for Fine-Tuned XTTS Checkpoints

Copies the best fine-tuned model weights to flowedit/Xtts/model.pth
and verifies deployment.

Usage:
    python deploy_finetuned.py \
        --finetune_dir ./finetune_output \
        --target_dir ../Xtts
"""

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path


def get_md5(file_path: Path) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096 * 1024), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Deploy fine-tuned XTTS model to FlowEdit")
    parser.add_argument("--finetune_dir", type=str, default="./finetune_output",
                        help="Path to fine-tuning output directory")
    parser.add_argument("--target_dir", type=str, default="../Xtts",
                        help="Target XTTS model directory in FlowEdit (default: ../Xtts)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Specific checkpoint file to deploy (default: auto-find best_model.pth or latest checkpoint)")
    args = parser.parse_args()

    finetune_dir = Path(args.finetune_dir).resolve()
    target_dir = Path(args.target_dir).resolve()

    if not finetune_dir.exists():
        print(f"Error: Fine-tune output directory does not exist: {finetune_dir}")
        sys.exit(1)

    if not target_dir.exists():
        print(f"Error: Target directory does not exist: {target_dir}")
        sys.exit(1)

    # 1. Locate model checkpoint
    src_ckpt = None
    if args.checkpoint:
        src_ckpt = Path(args.checkpoint).resolve()
        if not src_ckpt.exists():
            print(f"Error: Specified checkpoint does not exist: {src_ckpt}")
            sys.exit(1)
    else:
        # Auto-search best_model.pth or latest checkpoint_*.pth
        candidates = list(finetune_dir.glob("**/best_model.pth"))
        if not candidates:
            candidates = sorted(list(finetune_dir.glob("**/checkpoint_*.pth")), key=os.path.getmtime, reverse=True)

        if not candidates:
            # Look for any .pth file in finetune_dir
            candidates = list(finetune_dir.glob("**/*.pth"))

        if not candidates:
            print(f"Error: No .pth model checkpoints found inside {finetune_dir}")
            sys.exit(1)

        src_ckpt = candidates[0]

    print("=" * 60)
    print("DEPLOYING FINE-TUNED XTTS MODEL")
    print("=" * 60)
    print(f"  Source checkpoint: {src_ckpt}")
    print(f"  Target directory: {target_dir}")

    target_model_pth = target_dir / "model.pth"

    # Backup existing model.pth if present
    if target_model_pth.exists():
        backup_path = target_dir / "model.pth.bak"
        print(f"  Backing up existing model.pth -> {backup_path.name}")
        shutil.copy2(target_model_pth, backup_path)

    print(f"  Copying {src_ckpt.name} -> {target_model_pth}...")
    shutil.copy2(src_ckpt, target_model_pth)

    # Check MD5 and size
    size_mb = target_model_pth.stat().st_size / (1024 * 1024)
    print("\nComputing MD5 checksum to verify...")
    md5_hash = get_md5(target_model_pth)

    print("\n" + "=" * 60)
    print("DEPLOYMENT SUCCESSFUL!")
    print("=" * 60)
    print(f"  Deployed model: {target_model_pth}")
    print(f"  File size:      {size_mb:.2f} MB")
    print(f"  MD5 Checksum:   {md5_hash}")
    print("\nNext steps:")
    print("1. Restart your Uvicorn server on the server:")
    print("   fuser -k 8001/tcp")
    print("   uvicorn flowedit.api.main:app --host 0.0.0.0 --port 8001 --reload")
    print("2. Test pronunciation via /api/synthesize or /api/synthesize_raw!")
    print("=" * 60)


if __name__ == "__main__":
    main()
