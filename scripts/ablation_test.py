import sys
import os
import torch
import logging
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from flowedit.config import FlowEditConfig
from flowedit.backbone import create_backbone
from flowedit.alignment.whisper_aligner import WhisperAligner
from flowedit.optimizer.latent_optimizer import LatentOptimizer
from flowedit.utils.audio import AudioProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_ablation(text: str, target_word: str, ref_audio_path: str, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = FlowEditConfig()
    config.backbone.backbone_type = "xtts"
    config.optimization.n_steps = 50
    
    # 1. Setup Backbone
    logger.info("Loading XTTS Backbone...")
    bb = create_backbone(config.backbone)
    bb.load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bb.model.to(device)
    
    # 2. Setup Aligner
    logger.info("Loading Whisper Aligner...")
    aligner = WhisperAligner(config.alignment)
    aligner.load_model()
    aligner._model.to(device)
    
    # 3. Setup Optimizer
    logger.info("Setting up LatentOptimizer...")
    optimizer = LatentOptimizer(config.optimization, config.audio)
    audio_processor = AudioProcessor(config.audio)
    
    # Run Stage 1: Grounding
    logger.info(f"Aligning target word '{target_word}' in '{ref_audio_path}'")
    alignment = aligner.align(
        audio_path=ref_audio_path,
        target_word=target_word,
        full_text=target_word,
        language="en",
        ref_is_word_only=True
    )
    
    tokenizer = getattr(bb, "tokenizer", None) or getattr(bb.model, "tokenizer", None)
    alignment = aligner.map_to_token_indices(
        alignment=alignment,
        full_text=text,
        target_word=target_word,
        tokenizer=tokenizer,
        language="en"
    )
    
    if not alignment.token_indices:
        logger.error("Failed to map tokens!")
        return
        
    logger.info(f"Token indices: {alignment.token_indices}")
    
    speaker_conditioning = bb.get_speaker_embedding(ref_audio_path, "en", ref_text=target_word)
    
    # Step callback
    def on_step(step: int, loss_val: float, grad_val: float, perturbed_embeddings: torch.Tensor):
        if step % 10 == 0 or step == config.optimization.n_steps - 1:
            logger.info(f"Synthesizing audio for step {step}...")
            # Synthesize intermediate audio directly using the perturbed embeddings (no Hopfield)
            waveform, sr = bb.synthesize_from_embeddings(
                text_embeddings=perturbed_embeddings,
                speaker_conditioning=speaker_conditioning,
                text=text,
                language="en"
            )
            step_audio_path = output_path / f"step_{step}.wav"
            audio_processor.save_audio(waveform, str(step_audio_path), sr)
            logger.info(f"Saved: {step_audio_path}")
            
    # Run Stage 2: Optimization
    logger.info("Starting Optimization (Ablation Mode: No Hopfield)...")
    opt_result = optimizer.optimize(
        backbone=bb,
        text=text,
        ref_audio_path=ref_audio_path,
        token_indices=alignment.token_indices,
        speaker_conditioning=speaker_conditioning,
        language="en",
        target_word_start_time=alignment.start_time,
        target_word_end_time=alignment.end_time,
        progress_callback=on_step
    )
    
    import csv
    csv_path = output_path / "loss.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "grad_norm"])
        for i, (loss, grad) in enumerate(zip(opt_result.loss_history, opt_result.grad_norm_history)):
            writer.writerow([i, loss, grad])
            
    logger.info(f"Optimization finished! Final loss: {opt_result.final_loss:.4f}")
    logger.info(f"Saved metrics to: {csv_path}")
    logger.info(f"Ablation audio files saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlowEdit Optimization Ablation")
    parser.add_argument("--text", type=str, required=True, help="Full text")
    parser.add_argument("--word", type=str, required=True, help="Target word")
    parser.add_argument("--ref_audio", type=str, required=True, help="Reference audio path")
    parser.add_argument("--out_dir", type=str, default="ablation_results", help="Output directory")
    
    args = parser.parse_args()
    run_ablation(args.text, args.word, args.ref_audio, args.out_dir)
