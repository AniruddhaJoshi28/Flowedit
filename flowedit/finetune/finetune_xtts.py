"""
XTTS-v2 Fine-Tuning Script — Pure PyTorch (No Coqui Trainer)

Bypasses the Coqui Trainer entirely to avoid interface mismatches.
Uses a direct PyTorch training loop on the GPT component of XTTS.

Usage:
    CUDA_VISIBLE_DEVICES=0 python finetune/finetune_xtts.py \
        --dataset_dir ./formatted_dataset \
        --output_path ./finetune_output \
        --base_model_dir ./flowedit/Xtts \
        --epochs 50 \
        --lr 5e-6
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

# Default to GPU 0 on multi-GPU servers
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np


def load_wav(path, target_sr=22050):
    """Load a wav file, resample if needed, return mono float32 tensor."""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0)  # [T]


def main():
    parser = argparse.ArgumentParser(description="Fine-tune XTTS-v2 (Pure PyTorch)")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to formatted dataset (tts_train.csv, wavs/)")
    parser.add_argument("--output_path", type=str, default="./finetune_output",
                        help="Output directory for checkpoints")
    parser.add_argument("--base_model_dir", type=str, default="./flowedit/Xtts",
                        help="Path to XTTS model directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    base_model_dir = Path(args.base_model_dir).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Verify files
    train_csv = dataset_dir / "tts_train.csv"
    config_json = base_model_dir / "config.json"
    vocab_json = base_model_dir / "vocab.json"
    checkpoint_pth = base_model_dir / "base_model.pth"
    if not checkpoint_pth.exists():
        checkpoint_pth = base_model_dir / "model.pth"

    for f in [train_csv, config_json, vocab_json, checkpoint_pth]:
        if not f.exists():
            print(f"Error: Required file missing: {f}")
            sys.exit(1)

    # Parse training data from CSV
    samples = []
    with open(train_csv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("audio_file"):  # skip header
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                wav_path = dataset_dir / parts[0].strip()
                text = parts[1].strip()
                if wav_path.exists():
                    samples.append({"wav": str(wav_path), "text": text})

    if not samples:
        print("Error: No valid training samples found!")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("XTTS-v2 FINE-TUNING (Pure PyTorch)")
    print("=" * 60)
    print(f"  Device:         {device}")
    print(f"  Dataset:        {dataset_dir}")
    print(f"  Samples:        {len(samples)}")
    print(f"  Base checkpoint:{checkpoint_pth.name}")
    print(f"  Output:         {output_path}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Grad clip:      {args.grad_clip}")
    print("=" * 60)

    # ── Load XTTS Model ──────────────────────────────────────────
    print("\nLoading XTTS model...")
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    xtts_config = XttsConfig()
    xtts_config.load_json(str(config_json))

    model = Xtts.init_from_config(xtts_config)
    model.load_checkpoint(
        xtts_config,
        checkpoint_dir=str(base_model_dir),
        checkpoint_path=str(checkpoint_pth),
        vocab_path=str(vocab_json),
        eval=False,
        use_deepspeed=False,
    )
    model = model.to(device)
    print(f"  Model loaded on {device}")

    # ── Load DVAE (Required for training audio encoding) ─────────
    print("  Loading DVAE for audio tokenization...")
    from TTS.tts.layers.xtts.dvae import DiscreteVAE
    dvae = DiscreteVAE(
        channels=80,
        normalization=None,
        positional_dims=1,
        num_tokens=1024,
        codebook_dim=512,
        hidden_dim=512,
        num_resnet_blocks=3,
        kernel_size=3,
        num_layers=2,
        use_transposed_convs=False,
    )
    dvae_path = base_model_dir / "dvae.pth"
    if not dvae_path.exists():
        print(f"Error: {dvae_path} not found. DVAE is required for fine-tuning!")
        sys.exit(1)
    
    checkpoint = torch.load(str(dvae_path), map_location="cpu")
    if "model" in checkpoint:
        dvae.load_state_dict(checkpoint["model"], strict=False)
    else:
        dvae.load_state_dict(checkpoint, strict=False)
        
    dvae = dvae.to(device)
    dvae.eval()
    model.dvae = dvae

    # ── Freeze non-GPT parameters ────────────────────────────────
    print("  Freezing dVAE and HiFi-GAN vocoder...")
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze GPT only
    gpt = model.gpt
    for param in gpt.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in gpt.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} total")

    # ── Setup optimizer ──────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in gpt.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    # ── Preprocess all training samples ──────────────────────────
    print("\nPreprocessing training samples...")
    processed_samples = []
    for s in samples:
        wav = load_wav(s["wav"], target_sr=22050).to(device)
        text = s["text"]

        # Tokenize text using XTTS tokenizer
        text_tokens = torch.IntTensor(
            model.tokenizer.encode(text, lang="en")
        ).unsqueeze(0).to(device)

        # Get conditioning latents from the audio (speaker embedding)
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[s["wav"]],
            gpt_cond_len=model.config.gpt_cond_len if hasattr(model.config, "gpt_cond_len") else 6,
            gpt_cond_chunk_len=model.config.gpt_cond_chunk_len if hasattr(model.config, "gpt_cond_chunk_len") else 4,
        )

        # Convert audio to mel spectrogram for dvae encoding
        if not hasattr(model, 'mel_transform'):
            model.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=22050,
                n_fft=1024,
                win_length=1024,
                hop_length=256,
                f_min=0,
                f_max=8000,
                n_mels=80,
            ).to(device)
        
        mel = model.mel_transform(wav.unsqueeze(0))
        # Log-mel scaling (commonly used in Tortoise/XTTS)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        # Get discrete audio codes from dvae
        with torch.no_grad():
            codes = model.dvae.get_codebook_indices(mel)

        processed_samples.append({
            "text_tokens": text_tokens,
            "audio_codes": codes,
            "gpt_cond_latent": gpt_cond_latent,
            "speaker_embedding": speaker_embedding,
            "cond_mels": mel,
            "text": text,
        })
        print(f"  [OK] '{text}' -> {text_tokens.shape[1]} text tokens, {codes.shape[1]} audio codes")

    # ── Training Loop ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}")

    model.train()
    # Keep dvae and hifigan in eval mode
    if hasattr(model, "dvae"):
        model.dvae.eval()
    if hasattr(model, "hifigan_decoder"):
        model.hifigan_decoder.eval()

    best_loss = float("inf")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_start = time.time()

        for i, sample in enumerate(processed_samples):
            optimizer.zero_grad()

            text_tokens = sample["text_tokens"]
            audio_codes = sample["audio_codes"]
            gpt_cond_latent = sample["gpt_cond_latent"]
            cond_mels = sample["cond_mels"]
            # Get lengths
            text_lengths = torch.tensor([text_tokens.shape[1]], dtype=torch.long, device=device)
            audio_lengths = torch.tensor([audio_codes.shape[1]], dtype=torch.long, device=device)

            # Forward pass through GPT
            # Signature: (text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels=None, cond_idxs=None, cond_lens=None, cond_latents=None)
            res = model.gpt.forward(
                text_inputs=text_tokens,
                text_lengths=text_lengths,
                audio_codes=audio_codes,
                wav_lengths=audio_lengths,
                cond_mels=cond_mels,
                cond_latents=gpt_cond_latent,
            )

            # The forward pass typically returns (loss_text, loss_mel, _, _)
            if isinstance(res, tuple):
                loss_text = res[0]
                loss_mel = res[1]
                loss = loss_text + loss_mel
            elif isinstance(res, dict):
                loss = res.get("loss", sum(res.values()))
            else:
                loss = res


            loss.backward()

            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    gpt.parameters(), args.grad_clip
                )

            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(processed_samples), 1)
        elapsed = time.time() - epoch_start

        print(f"  Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = output_path / "best_model.pth"
            torch.save(model.state_dict(), best_path)

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = output_path / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"    -> Saved checkpoint: {ckpt_path.name}")

    # Save final model
    final_path = output_path / "final_model.pth"
    torch.save(model.state_dict(), final_path)

    print(f"\n{'='*60}")
    print("FINE-TUNING COMPLETE!")
    print(f"{'='*60}")
    print(f"  Best loss:       {best_loss:.6f}")
    print(f"  Best model:      {output_path / 'best_model.pth'}")
    print(f"  Final model:     {final_path}")
    print(f"  Checkpoints in:  {output_path}")
    print(f"\nRun deploy_finetuned.py to update FlowEdit's active model!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
