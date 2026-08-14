"""
Script to verify if model.pth is genuinely fine-tuned compared to base_model.pth.
Loads both checkpoints, iterates through parameters, and computes differences.
"""

import torch
import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Verify if XTTS model is fine-tuned")
    parser.add_argument("--base", type=str, default="flowedit/Xtts/base_model.pth", help="Path to base model")
    parser.add_argument("--finetuned", type=str, default="flowedit/Xtts/model.pth", help="Path to fine-tuned model")
    args = parser.parse_args()

    base_path = Path(args.base).resolve()
    finetuned_path = Path(args.finetuned).resolve()

    if not base_path.exists():
        print(f"Error: Base model not found at {base_path}")
        return
    if not finetuned_path.exists():
        print(f"Error: Fine-tuned model not found at {finetuned_path}")
        return

    print(f"Loading base model from: {base_path}")
    base_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    base_state = base_ckpt.get("model", base_ckpt.get("state_dict", base_ckpt))
    
    print(f"Loading fine-tuned model from: {finetuned_path}")
    finetuned_ckpt = torch.load(finetuned_path, map_location="cpu", weights_only=False)
    finetuned_state = finetuned_ckpt.get("model", finetuned_ckpt.get("state_dict", finetuned_ckpt))

    print("-" * 80)
    print(f"{'Layer Name':<50} | {'Mean Abs Diff':<15} | {'Max Abs Diff':<15} | Changed?")
    print("-" * 80)

    changed_count = 0
    total_count = 0

    for key in base_state.keys():
        if key not in finetuned_state:
            print(f"{key[:48]:<50} | {'MISSING IN FINETUNED':<40}")
            continue
        
        total_count += 1
        base_tensor = base_state[key].float()
        finetuned_tensor = finetuned_state[key].float()

        if base_tensor.shape != finetuned_tensor.shape:
            print(f"{key[:48]:<50} | {'SHAPE MISMATCH':<40}")
            continue

        diff = torch.abs(base_tensor - finetuned_tensor)
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()

        is_changed = max_diff > 1e-6
        if is_changed:
            changed_count += 1

        # Only print changed layers to keep output clean, unless it's a small model
        if is_changed or total_count <= 5:
            changed_str = "Yes" if is_changed else "No"
            print(f"{key[:48]:<50} | {mean_diff:<15.6e} | {max_diff:<15.6e} | {changed_str}")

    print("-" * 80)
    print(f"Parameters changed : {changed_count} / {total_count}")
    
    if changed_count > 0:
        print("Result: model.pth HAS been fine-tuned (weights are different from base_model.pth).")
    else:
        print("Result: model.pth is IDENTICAL to base_model.pth (not fine-tuned).")

if __name__ == "__main__":
    main()
