"""
FlowEdit CLI: Learn a pronunciation correction.

Usage:
    python scripts/correct.py \
        --text "My friend Siobhan is visiting" \
        --target-word "Siobhan" \
        --ref-audio ./reference/siobhan_correct.wav \
        --memory-path ./corrections.pt

    # With speaker reference (optional)
    python scripts/correct.py \
        --text "The city of Llanfairpwllgwyngyll" \
        --target-word "Llanfairpwllgwyngyll" \
        --ref-audio ./reference/llanfair.wav \
        --speaker-wav ./speaker.wav \
        --memory-path ./corrections.pt \
        --language cy
"""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowedit.config import FlowEditConfig
from flowedit.pipeline.correction_loop import CorrectionLoop


def main():
    parser = argparse.ArgumentParser(
        description="FlowEdit: Learn a pronunciation correction from reference audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Learn "Siobhan" pronunciation:
    python correct.py --text "My friend Siobhan is visiting" \\
                      --target-word "Siobhan" \\
                      --ref-audio ./siobhan.wav

  Learn with custom memory path:
    python correct.py --text "Welcome to Nguyen's restaurant" \\
                      --target-word "Nguyen" \\
                      --ref-audio ./nguyen.wav \\
                      --memory-path ./my_corrections.pt
        """,
    )

    # Required arguments
    parser.add_argument(
        "--text", required=True,
        help="Full text containing the target word"
    )
    parser.add_argument(
        "--target-word", required=True,
        help="The word to correct pronunciation of"
    )
    parser.add_argument(
        "--ref-audio", required=True,
        help="Path to reference audio with correct pronunciation (≥1.5s)"
    )

    # Optional arguments
    parser.add_argument(
        "--memory-path", default="./corrections.pt",
        help="Path to save/load Hopfield memory (default: ./corrections.pt)"
    )
    parser.add_argument(
        "--speaker-wav", default=None,
        help="Speaker reference audio (defaults to ref-audio)"
    )
    parser.add_argument(
        "--language", default="en",
        help="Language code (default: en)"
    )

    # Optimization overrides
    parser.add_argument(
        "--n-steps", type=int, default=50,
        help="Number of optimization steps (default: 50)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01,
        help="Initial learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--lambda-reg", type=float, default=0.001,
        help="Regularization weight (default: 0.001)"
    )
    parser.add_argument(
        "--use-f0-loss", action="store_true",
        help="Enable F0 loss for tonal languages"
    )

    # General
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device to use (default: cuda)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build config with overrides
    config = FlowEditConfig()
    config.optimization.n_steps = args.n_steps
    config.optimization.lr_start = args.lr
    config.optimization.lambda_reg = args.lambda_reg
    config.optimization.use_f0_loss = args.use_f0_loss
    config.optimization.f0_loss_alpha = 0.3 if args.use_f0_loss else 0.0
    config.backbone.device = args.device

    # Initialize and load models
    loop = CorrectionLoop(config)
    loop.load_models(memory_path=args.memory_path)

    # Run correction
    result = loop.correct(
        text=args.text,
        target_word=args.target_word,
        ref_audio_path=args.ref_audio,
        speaker_wav=args.speaker_wav,
        language=args.language,
    )

    # Report results
    if result.success:
        print(f"\n✅ Correction successful!")
        print(f"   Word: {result.word}")
        print(f"   Time: {result.wall_clock_seconds:.1f}s")
        print(f"   Final loss: {result.optimization.final_loss:.4f}")
        print(f"   Converged: {result.optimization.converged}")
        print(f"   Memory: {result.memory_size} corrections stored")

        # Save memory
        loop.save_memory(args.memory_path)
        print(f"   Saved to: {args.memory_path}")
    else:
        print(f"\n❌ Correction failed: {result.error_message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
