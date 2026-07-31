"""
Phase 0: Proof-of-Concept — Validate Embedding Hook Consistency

This script verifies a critical assumption before implementing the full
XTTS backend for FlowEdit:

  "The same δ perturbation injected via gpt.text_embedding hook
   affects BOTH the teacher-forced forward path AND the autoregressive
   inference path."

If both tests pass, the teacher-forced optimization of δ will transfer
correctly to inference. If not, we need to find a different hook location.

Usage (on the server):
    cd ~/projects/flowedit
    source .venv/bin/activate   # or: conda activate xtts_api
    python -m flowedit.validate_hook_paths
"""

import torch
import torch.nn.functional as F
import os
import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("Phase 0: Validate XTTS Embedding Hook Paths")
    logger.info("=" * 70)

    # ─── Step 1: Load XTTS Model ─────────────────────────────────────────
    logger.info("\n[1/5] Loading XTTS model...")

    from flowedit.config import FlowEditConfig
    config = FlowEditConfig()

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    model_dir = config.backbone.xtts_model_dir
    checkpoint_path = os.path.join(model_dir, config.backbone.xtts_checkpoint)
    config_path = os.path.join(model_dir, "config.json")
    vocab_path = os.path.join(model_dir, "vocab.json")

    xtts_config = XttsConfig()
    xtts_config.load_json(config_path)
    model = Xtts.init_from_config(xtts_config)
    model.load_checkpoint(
        xtts_config,
        checkpoint_dir=model_dir,
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        eval=True,
        use_deepspeed=False,
        strict=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    gpt = model.gpt
    text_embedding_layer = gpt.text_embedding  # nn.Embedding(num_text_tokens, 1024)
    logger.info(f"  text_embedding: {text_embedding_layer}")
    logger.info(f"  model_dim: {gpt.model_dim}")
    logger.info(f"  Device: {device}")

    # ─── Step 2: Prepare test inputs ─────────────────────────────────────
    logger.info("\n[2/5] Preparing test inputs...")

    test_text = "Hello, my name is Mrunmayee."
    language = "en"

    # Tokenize
    tokens = model.tokenizer.encode(test_text, lang=language)
    if isinstance(tokens, list):
        if isinstance(tokens[0], list):
            tokens = tokens[0]
    tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    logger.info(f"  Text: '{test_text}'")
    logger.info(f"  Tokens shape: {tokens_tensor.shape}")

    # Get base embeddings (no perturbation)
    with torch.no_grad():
        base_embeddings = text_embedding_layer(tokens_tensor)
    logger.info(f"  Base embeddings shape: {base_embeddings.shape}")
    logger.info(f"  Base embeddings norm: {torch.norm(base_embeddings).item():.4f}")

    # Create random δ perturbation (significant magnitude to see effect)
    delta = torch.randn_like(base_embeddings) * 0.5
    perturbed_embeddings = base_embeddings + delta
    logger.info(f"  δ norm: {torch.norm(delta).item():.4f}")

    # ─── Step 3: Test teacher-forced forward ─────────────────────────────
    logger.info("\n[3/5] TEST A: Teacher-forced forward path...")
    logger.info("  Checking if δ changes the GPT logits during training forward...")

    # We need a dummy audio code sequence for teacher forcing
    # Use the GPT's forward method which expects text_tokens + audio_codes
    # First, let's check what methods are available
    logger.info(f"  GPT class: {type(gpt).__name__}")
    logger.info(f"  GPT methods: {[m for m in dir(gpt) if not m.startswith('_') and callable(getattr(gpt, m))]}")

    # Check if GPT has a forward/compute_embeddings method
    has_compute_embeddings = hasattr(gpt, 'compute_embeddings')
    has_forward = hasattr(gpt, 'forward')
    logger.info(f"  has compute_embeddings: {has_compute_embeddings}")
    logger.info(f"  has forward: {has_forward}")

    # Create a hook that replaces embeddings with perturbed version
    hook_activated = {"count": 0, "output_shape": None}

    def make_hook(replacement_embeddings):
        def hook_fn(module, inputs, output):
            hook_activated["count"] += 1
            hook_activated["output_shape"] = output.shape
            # Replace output with perturbed embeddings
            # Match shapes: output may be [B, T, D], replacement may be [1, T', D]
            out = output.clone()
            T_out = output.shape[1]
            T_rep = replacement_embeddings.shape[1]
            T = min(T_out, T_rep)
            out[:, :T, :] = replacement_embeddings[:, :T, :].to(
                device=output.device, dtype=output.dtype
            )
            return out
        return hook_fn

    # ── Test A1: Forward WITHOUT hook ──
    # We need to find the right way to call GPT forward
    # Let's try with a simple approach: just check if the embedding layer
    # is called during both paths

    logger.info("\n  --- A1: Baseline forward (no hook) ---")
    hook_activated["count"] = 0

    # Register a monitoring hook (doesn't change output, just observes)
    monitor_outputs = []
    def monitor_hook(module, inputs, output):
        monitor_outputs.append(output.clone().detach())
    
    h = text_embedding_layer.register_forward_hook(monitor_hook)

    # Try calling model.inference to see if text_embedding is triggered
    # First, we need speaker conditioning
    # Use a simple test: just generate a short dummy conditioning
    logger.info("  Getting speaker conditioning (using dummy/default)...")

    # Try to find any WAV file for conditioning
    speaker_wav = None
    speakers_dir = os.path.join(os.path.dirname(model_dir), "speakers")
    if os.path.isdir(speakers_dir):
        for f in os.listdir(speakers_dir):
            if f.endswith(".wav"):
                speaker_wav = os.path.join(speakers_dir, f)
                break

    # Also check for a resources/default_speaker.wav
    if not speaker_wav:
        resources_dir = os.path.join(os.path.dirname(model_dir), "resources")
        if os.path.isdir(resources_dir):
            for f in os.listdir(resources_dir):
                if f.endswith(".wav"):
                    speaker_wav = os.path.join(resources_dir, f)
                    break

    if not speaker_wav:
        logger.warning(
            "  No speaker WAV found! Creating a dummy sine wave..."
        )
        import tempfile
        import soundfile as sf

        sr = 22050
        t = np.linspace(0, 2.0, int(sr * 2.0))
        dummy_audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        fd, speaker_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(speaker_wav, dummy_audio, sr)
        logger.info(f"  Created dummy WAV: {speaker_wav}")
    else:
        logger.info(f"  Using speaker WAV: {speaker_wav}")

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[speaker_wav],
        gpt_cond_len=6,
        gpt_cond_chunk_len=4,
        max_ref_length=10,
    )
    logger.info(f"  gpt_cond_latent shape: {gpt_cond_latent.shape}")
    logger.info(f"  speaker_embedding shape: {speaker_embedding.shape}")

    h.remove()
    monitor_outputs.clear()

    # ── Test A2: Inference WITHOUT perturbation ──
    logger.info("\n  --- A2: Inference WITHOUT δ (baseline) ---")

    with torch.no_grad():
        out_baseline = model.inference(
            text=test_text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.01,  # Near-deterministic for reproducibility
            top_k=1,           # Greedy decoding
            top_p=1.0,
            enable_text_splitting=False,
        )
    wav_baseline = out_baseline["wav"]
    if isinstance(wav_baseline, torch.Tensor):
        wav_baseline = wav_baseline.cpu().numpy()
    else:
        wav_baseline = np.array(wav_baseline)
    logger.info(f"  Baseline waveform: {wav_baseline.shape}, "
                f"mean={wav_baseline.mean():.6f}, std={wav_baseline.std():.6f}")

    # ── Test A3: Inference WITH δ (hooked) ──
    logger.info("\n  --- A3: Inference WITH δ (embedding hook active) ---")

    hook_activated["count"] = 0
    h_perturb = text_embedding_layer.register_forward_hook(
        make_hook(perturbed_embeddings)
    )

    with torch.no_grad():
        out_perturbed = model.inference(
            text=test_text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.01,
            top_k=1,
            top_p=1.0,
            enable_text_splitting=False,
        )

    h_perturb.remove()

    wav_perturbed = out_perturbed["wav"]
    if isinstance(wav_perturbed, torch.Tensor):
        wav_perturbed = wav_perturbed.cpu().numpy()
    else:
        wav_perturbed = np.array(wav_perturbed)

    logger.info(f"  Hook activated {hook_activated['count']} time(s)")
    logger.info(f"  Hook output shape: {hook_activated['output_shape']}")
    logger.info(f"  Perturbed waveform: {wav_perturbed.shape}, "
                f"mean={wav_perturbed.mean():.6f}, std={wav_perturbed.std():.6f}")

    # Compare
    min_len = min(len(wav_baseline), len(wav_perturbed))
    if min_len > 0:
        diff = np.abs(wav_baseline[:min_len] - wav_perturbed[:min_len])
        max_diff = diff.max()
        mean_diff = diff.mean()
        length_diff = abs(len(wav_baseline) - len(wav_perturbed))
    else:
        max_diff = 0.0
        mean_diff = 0.0
        length_diff = 0

    logger.info(f"\n  INFERENCE COMPARISON:")
    logger.info(f"    Baseline length:  {len(wav_baseline)} samples")
    logger.info(f"    Perturbed length: {len(wav_perturbed)} samples")
    logger.info(f"    Length diff:       {length_diff} samples")
    logger.info(f"    Max sample diff:   {max_diff:.6f}")
    logger.info(f"    Mean sample diff:  {mean_diff:.6f}")

    inference_hook_works = (
        hook_activated["count"] > 0 and
        (max_diff > 0.01 or length_diff > 100)
    )

    # ─── Step 4: Test teacher-forced forward (differentiability) ──────────
    logger.info("\n[4/5] TEST B: Teacher-forced forward differentiability...")
    logger.info("  Checking if gradients flow from logits back to δ...")

    # Create δ with requires_grad
    delta_opt = torch.randn(
        1, tokens_tensor.shape[1], gpt.model_dim,
        device=device, dtype=torch.float32,
        requires_grad=True,
    ) * 0.1

    perturbed_for_grad = base_embeddings.detach().float() + delta_opt

    # Hook to inject perturbed embeddings
    def grad_hook(module, inputs, output):
        out = perturbed_for_grad.to(device=output.device, dtype=output.dtype)
        # Pad or truncate to match output shape
        T_out = output.shape[1]
        T_in = out.shape[1]
        if T_in < T_out:
            pad = output[:, T_in:, :].clone()
            return torch.cat([out, pad], dim=1)
        return out[:, :T_out, :]

    h_grad = text_embedding_layer.register_forward_hook(grad_hook)

    teacher_force_works = False
    grad_nonzero = False

    try:
        # Try to find the GPT's training forward method
        # XTTS GPT.forward expects: (text_tokens, text_lengths, audio_codes, wav_lengths, ...)
        # We'll create dummy audio codes

        # Check the forward signature
        import inspect
        fwd_sig = inspect.signature(gpt.forward)
        logger.info(f"  GPT.forward signature: {fwd_sig}")

        # Create dummy audio codes (just zeros — we only care about gradient flow)
        n_audio_tokens = 50  # Short sequence
        dummy_audio_codes = torch.randint(
            0, gpt.num_audio_tokens - 2,  # Avoid start/stop tokens
            (1, n_audio_tokens),
            device=device, dtype=torch.long,
        )

        text_lengths = torch.tensor([tokens_tensor.shape[1]], device=device)
        wav_lengths = torch.tensor([n_audio_tokens], device=device)

        # Try calling forward
        try:
            output = gpt(
                text_tokens=tokens_tensor,
                text_lengths=text_lengths,
                audio_codes=dummy_audio_codes,
                wav_lengths=wav_lengths,
                cond_latents=gpt_cond_latent,
            )

            # output should be a loss or logits
            if isinstance(output, dict):
                logger.info(f"  Forward returned dict keys: {output.keys()}")
                loss = output.get("loss", None)
                if loss is None and "logits" in output:
                    logits = output["logits"]
                    loss = logits.sum()
            elif isinstance(output, tuple):
                logger.info(f"  Forward returned tuple of {len(output)} elements")
                loss = output[0] if output[0].requires_grad else output[0].sum()
            elif isinstance(output, torch.Tensor):
                logger.info(f"  Forward returned tensor: {output.shape}")
                loss = output.sum() if output.dim() > 0 else output
            else:
                logger.info(f"  Forward returned: {type(output)}")
                loss = None

            if loss is not None and loss.requires_grad:
                loss.backward()
                if delta_opt.grad is not None:
                    grad_norm = torch.norm(delta_opt.grad).item()
                    grad_nonzero = grad_norm > 1e-10
                    logger.info(f"  ✓ Gradient computed! ||∇_δ|| = {grad_norm:.6f}")
                else:
                    logger.info("  ✗ delta_opt.grad is None (gradient didn't reach δ)")
            else:
                logger.info(f"  ✗ Could not compute loss (loss={loss})")

            teacher_force_works = True

        except Exception as e:
            logger.warning(f"  GPT.forward() call failed: {e}")
            logger.info("  Trying alternative forward signatures...")

            # Try without cond_latents
            try:
                delta_opt.grad = None if delta_opt.grad is not None else None
                delta_opt2 = torch.randn_like(delta_opt, requires_grad=True) * 0.1
                perturbed2 = base_embeddings.detach().float() + delta_opt2

                # Some versions use compute_embeddings + transformer directly
                if hasattr(gpt, 'compute_embeddings'):
                    logger.info("  Trying gpt.compute_embeddings()...")
                    embs = gpt.compute_embeddings(
                        cond_latents=gpt_cond_latent,
                        text_inputs=tokens_tensor,
                    )
                    logger.info(f"  compute_embeddings returned: {type(embs)}, shape={embs.shape if hasattr(embs, 'shape') else 'N/A'}")

            except Exception as e2:
                logger.warning(f"  Alternative forward also failed: {e2}")

    finally:
        h_grad.remove()

    # ─── Step 5: Summary ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\n  TEST A — Inference Hook:")
    logger.info(f"    Hook triggered during inference:  {'✅ YES' if hook_activated['count'] > 0 else '❌ NO'}")
    logger.info(f"    δ changes inference output:       {'✅ YES' if inference_hook_works else '❌ NO'}")
    logger.info(f"    Hook trigger count:               {hook_activated['count']}")

    logger.info(f"\n  TEST B — Teacher-Forced Forward:")
    logger.info(f"    GPT.forward() callable:           {'✅ YES' if teacher_force_works else '❌ NO'}")
    logger.info(f"    Gradients reach δ:                {'✅ YES' if grad_nonzero else '❌ NO'}")

    logger.info(f"\n  OVERALL VERDICT:")
    if inference_hook_works and grad_nonzero:
        logger.info("    ✅✅ BOTH PATHS CONFIRMED — proceed with full implementation!")
        logger.info("    The same gpt.text_embedding hook works for both optimization")
        logger.info("    (teacher-forced, differentiable) and inference (autoregressive).")
    elif inference_hook_works and not grad_nonzero:
        logger.info("    ⚠️  Inference hook works, but teacher-forced gradients need investigation.")
        logger.info("    Consider: direct embedding substitution instead of hook-based injection,")
        logger.info("    or zeroth-order optimization (finite differences) as fallback.")
    elif not inference_hook_works and grad_nonzero:
        logger.info("    ⚠️  Teacher-forced path works, but inference hook doesn't affect output.")
        logger.info("    The inference path may bypass gpt.text_embedding.")
        logger.info("    Need to find the correct hook location for inference.")
    else:
        logger.info("    ❌ Neither path confirmed — need deeper investigation.")
        logger.info("    Check if XTTS GPT uses a different embedding mechanism.")

    logger.info("\n" + "=" * 70)

    return inference_hook_works, grad_nonzero


if __name__ == "__main__":
    main()
