import os
import json
import logging
import subprocess
import shutil
from typing import List, Dict, Union, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class F5TTSFinetuner:
    """Wrapper for the official F5-TTS training pipeline.
    
    Instead of writing a custom PyTorch training loop that might diverge from
    the official implementation (e.g. EMA, mixed precision, checkpoints, optimizers),
    this wrapper converts our standard dataset format into the F5-TTS expected format
    (JSONL metadata + wavs) and launches the official accelerate training script.
    """
    def __init__(
        self,
        exp_name: str,
        output_dir: str = "checkpoints/f5tts_finetune",
        accelerate_config: Optional[str] = None
    ):
        self.exp_name = exp_name
        self.output_dir = Path(output_dir) / exp_name
        self.accelerate_config = accelerate_config
        self.dataset_dir = self.output_dir / "dataset"
        
    def prepare_dataset(self, audio_text_pairs: List[Dict[str, str]]):
        """Converts FlowEdit audio/text pairs into F5-TTS JSONL format.
        
        Expected input format:
        [
            {"audio_path": "/path/to/audio1.wav", "text": "This is sentence one."},
            {"audio_path": "/path/to/audio2.wav", "text": "Another sentence."}
        ]
        """
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        wavs_dir = self.dataset_dir / "wavs"
        wavs_dir.mkdir(exist_ok=True)
        
        metadata_path = self.dataset_dir / "metadata.jsonl"
        
        logger.info(f"Preparing dataset for F5-TTS. Saving to {self.dataset_dir}")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for i, pair in enumerate(audio_text_pairs):
                src_audio = pair["audio_path"]
                text = pair["text"]
                
                # Copy audio to dataset directory
                ext = Path(src_audio).suffix
                dst_audio = wavs_dir / f"sample_{i:04d}{ext}"
                shutil.copy2(src_audio, dst_audio)
                
                # F5-TTS expects {"audio_file": ..., "text": ...}
                record = {
                    "audio_file": str(dst_audio.absolute()),
                    "text": text
                }
                f.write(json.dumps(record) + "\n")
                
        logger.info(f"Prepared {len(audio_text_pairs)} samples.")
        return str(metadata_path)
        
    def train(
        self,
        learning_rate: float = 1e-5,
        batch_size: int = 4,
        epochs: int = 10,
        use_lora: bool = True,
        lora_rank: int = 16,
        save_steps: int = 500
    ):
        """Launches the official F5-TTS accelerate training script."""
        
        metadata_path = self.dataset_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset not found at {metadata_path}. Call prepare_dataset() first.")
            
        logger.info("Launching F5-TTS official training pipeline...")
        
        # Build the base accelerate command
        cmd = ["accelerate", "launch"]
        if self.accelerate_config:
            cmd.extend(["--config_file", self.accelerate_config])
            
        # The official training script. We assume standard module launch.
        # This resolves to python -m f5_tts.train.finetune_cli
        cmd.extend(["-m", "f5_tts.train.finetune_cli"])
        
        # Add dataset and output arguments
        cmd.extend([
            "--dataset_file", str(metadata_path),
            "--output_dir", str(self.output_dir),
            "--exp_name", self.exp_name,
            "--learning_rate", str(learning_rate),
            "--batch_size", str(batch_size),
            "--epochs", str(epochs),
            "--save_steps", str(save_steps)
        ])
        
        # LoRA vs Full Fine-tuning
        if use_lora:
            cmd.extend(["--use_lora", "true", "--lora_rank", str(lora_rank)])
            logger.info(f"Mode: LoRA Fine-Tuning (Rank {lora_rank})")
        else:
            cmd.extend(["--use_lora", "false"])
            logger.info("Mode: Full Fine-Tuning")
            
        logger.info(f"Executing: {' '.join(cmd)}")
        
        try:
            # We use subprocess.Popen to stream the output to the console
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in process.stdout:
                print(line, end="")
                
            process.wait()
            
            if process.returncode != 0:
                raise RuntimeError(f"F5-TTS training failed with exit code {process.returncode}")
                
            logger.info(f"Training completed successfully. Checkpoints saved in {self.output_dir}")
            
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")
            process.terminate()
            
        return str(self.output_dir)
