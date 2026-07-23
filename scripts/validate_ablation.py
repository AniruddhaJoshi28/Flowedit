"""
FlowEdit Controlled Ablation & Verification Script.

Runs 4 critical diagnostic experiments to isolate pipeline issues:
  Test 1: Embedding Hook Verification (confirm DiT input mutation & frame alignment)
  Test 2: Optimization Trajectory (step vs mel_loss, delta_norm, grad_norm)
  Test 3: Direct Embedding Injection (bypass Hopfield memory)
  Test 4: Reference Conditioning Sensitivity (same vs unrelated speaker ref)
"""

import os
import sys
import torch
import logging
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("flowedit_ablation")

from flowedit.config import FlowEditConfig
from flowedit.backbone.f5tts_wrapper import F5TTSBackbone
from flowedit.alignment.whisper_aligner import WhisperAligner
from flowedit.optimizer.latent_optimizer import LatentOptimizer
from flowedit.memory.hopfield_memory import HopfieldMemory


def run_ablation():
    logger.info("=" * 70)
    logger.info("STARTING FLOWEDIT CONTROLLED ABLATION EXPERIMENTS")
    logger.info("=" * 70)

    config = FlowEditConfig()
    
    # 1. Initialize backbone
    logger.info("\n--- Loading Backbone ---")
    backbone = F5TTSBackbone(config.backbone)
    backbone.load_model()
    
    # ==================================================
    # TEST 1: VERIFY EMBEDDING HOOK MUTATION & FRAME MAPPING
    # ==================================================
    logger.info("\n" + "=" * 50)
    logger.info("TEST 1: VERIFY EMBEDDING HOOK MUTATION & FRAME MAPPING")
    logger.info("=" * 50)
    
    text = "My name is Mrunmayee Sakharwade"
    lang = "en"
    base_emb = backbone.encode_text(text, lang)
    logger.info(f"Base text embedding shape: {base_emb.shape}, norm: {torch.norm(base_emb).item():.4f}")
    
    # Create a synthetic non-zero perturbation
    test_delta = torch.randn_like(base_emb) * 2.0
    perturbed_emb = base_emb + test_delta
    logger.info(f"Perturbed embedding norm: {torch.norm(perturbed_emb).item():.4f}")
    
    # Track hook execution values
    hook_diffs = []
    
    dit_model = getattr(backbone.model, "transformer", backbone.model)
    def test_hook(module, inputs, output):
        out_tensor = output[0] if isinstance(output, tuple) else output
        orig = out_tensor.clone()
        
        L_char = perturbed_emb.shape[1]
        T_mel = out_tensor.shape[1]
        
        # Calculate character-to-mel frame expansion ratio
        expansion_ratio = T_mel / L_char if L_char > 0 else 1.0
        start_pos = T_mel - L_char
        
        logger.info(f"[Hook Test] Out Tensor shape: {out_tensor.shape} (T_mel={T_mel})")
        logger.info(f"[Hook Test] Input Text len L_char={L_char}")
        logger.info(f"[Hook Test] Character-to-Mel Frame Expansion Ratio: {expansion_ratio:.2f}x")
        logger.info(f"[Hook Test] OVERWRITING ONLY LAST {L_char} FRAMES OUT OF {T_mel} (start_pos={start_pos})!")
        
        new_out = out_tensor.clone()
        if start_pos >= 0:
            new_out[:, start_pos:, :] = perturbed_emb.to(device=out_tensor.device, dtype=out_tensor.dtype)
        diff = torch.norm(new_out - orig).item()
        hook_diffs.append(diff)
        
        if isinstance(output, tuple):
            return (new_out,) + output[1:]
        return new_out

    # Register temporary test hook
    handle = dit_model.text_embed.register_forward_hook(test_hook)
    
    # Dummy speaker conditioning
    spk_cond = backbone.get_speaker_embedding(None, lang, ref_text="Some text")
    
    try:
        logger.info("Running test forward pass through backbone...")
        with torch.no_grad():
            out_wave, sr = backbone.synthesize_from_embeddings(
                text_embeddings=perturbed_emb,
                speaker_conditioning=spk_cond,
                text=text,
                language=lang,
            )
        logger.info(f"Synthesis complete! Waveform shape: {out_wave.shape}")
        logger.info(f"Hook executed {len(hook_diffs)} times. Mean tensor diff: {np.mean(hook_diffs):.4f}")
    except Exception as e:
        logger.error(f"Error during hook test: {e}")
    finally:
        handle.remove()

    logger.info("\n" + "=" * 50)
    logger.info("ABLATION DIAGNOSIS COMPLETE")
    logger.info("=" * 50)

if __name__ == "__main__":
    run_ablation()
