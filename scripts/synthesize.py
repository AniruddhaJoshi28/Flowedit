"""
FlowEdit CLI: Synthesize speech with pronunciation corrections.

Usage:
    python scripts/synthesize.py \
        --text "Siobhan's presentation was excellent" \
        --speaker-wav ./speaker.wav \
        --memory-path ./corrections.pt \
        --output ./output.wav

    # Compare corrected vs baseline
    python scripts/synthesize.py \
        --text "Siobhan's presentation was excellent" \
        --speaker-wav ./speaker.wav \
        --memory-path ./corrections.pt \
        --compare \
        --output-dir ./comparison/
"""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowedit.config import FlowEditConfig
from flowedit.pipeline.inference import FlowEditInference


def main():
    parser = argparse.ArgumentParser(
        description="FlowEdit: Synthesize speech with pronunciation corrections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic synthesis:
    python synthesize.py --text "Hello Siobhan" \\
                         --speaker-wav ./speaker.wav \\
                         --output ./output.wav

  Compare corrected vs baseline:
    python synthesize.py --text "Hello Siobhan" \\
                         --speaker-wav ./speaker.wav \\
                         --compare --output-dir ./comparison/

  List stored corrections:
    python synthesize.py --list-corrections --memory-path ./corrections.pt
        """,
    )

    # Synthesis arguments
    parser.add_argument(
        "--text",
        help="Text to synthesize"
    )
    parser.add_argument(
        "--speaker-wav", 
        help="Speaker reference audio for voice cloning (optional, defaults to preset voice)"
    )
    parser.add_argument(
        "--speaker-name", default="female",
        help="Preset speaker voice name if --speaker-wav is not provided: 'female' (Blessing) or 'male' (Michael)"
    )
    parser.add_argument(
        "--output", default="./output.wav",
        help="Output audio file path (default: ./output.wav)"
    )

    # Memory
    parser.add_argument(
        "--memory-path", default="./corrections.pt",
        help="Path to Hopfield memory file (default: ./corrections.pt)"
    )

    # Options
    parser.add_argument(
        "--language", default="en",
        help="Language code (default: en)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Generate both corrected and baseline audio"
    )
    parser.add_argument(
        "--output-dir", default="./comparison",
        help="Output directory for comparison mode"
    )
    parser.add_argument(
        "--gate-info", action="store_true",
        help="Print gate activation details"
    )

    # Utility
    parser.add_argument(
        "--list-corrections", action="store_true",
        help="List all stored corrections and exit"
    )

    # General
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device (default: cuda)"
    )

    args = parser.parse_args()

    # Pre-process text to reduce hallucinations
    if args.text:
        # Add trailing punctuation if missing (helps DiT/autoregressive models stop cleanly)
        if not args.text.strip().endswith(('.', '!', '?')):
            args.text = args.text.strip() + '.'

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build config (XTTS-v2 backbone by default)
    config = FlowEditConfig()
    config.backbone.backbone_type = "xtts"
    config.backbone.device = args.device

    # Initialize inference pipeline
    inference = FlowEditInference(config)
    inference.load(memory_path=args.memory_path)

    # Handle list-corrections mode
    if args.list_corrections:
        corrections = inference.memory.list_corrections()
        if not corrections:
            print("No corrections stored.")
        else:
            print(f"\n📋 Stored corrections ({len(corrections)}):")
            print(f"{'#':<4} {'Word':<20} {'Updates':<8} {'Key ‖·‖':<10} {'Value ‖·‖':<10}")
            print("-" * 52)
            for c in corrections:
                print(
                    f"{c['index']:<4} {c['word']:<20} {c['update_count']:<8} "
                    f"{c['key_norm']:<10.4f} {c['value_norm']:<10.4f}"
                )
        return

    # Validate required args for synthesis
    if not args.text:
        parser.error("--text is required for synthesis")

    if args.compare:
        # Comparison mode: generate both versions
        result = inference.compare(
            text=args.text,
            speaker_wav=args.speaker_wav,
            language=args.language,
            output_dir=args.output_dir,
            speaker_name=args.speaker_name,
        )

        corrected = result["corrected"]
        baseline = result["baseline"]

        print(f"\n📊 Comparison Results:")
        print(f"   Text: \"{args.text}\"")
        print(f"   Corrections applied: {corrected.get('corrections_applied', 0)} tokens")
        print(f"   Corrected inference: {corrected.get('inference_time_ms', 0):.0f}ms")
        print(f"   Baseline inference:  {baseline.get('inference_time_ms', 0):.0f}ms")
        print(f"   Saved to: {args.output_dir}/")

    else:
        # Standard synthesis
        result = inference.synthesize(
            text=args.text,
            speaker_wav=args.speaker_wav,
            language=args.language,
            output_path=args.output,
            return_gate_info=args.gate_info,
            speaker_name=args.speaker_name,
        )

        print(f"\n🔊 Synthesis complete!")
        print(f"   Text: \"{args.text}\"")
        print(f"   Corrections applied: {result['corrections_applied']} tokens")
        print(f"   Inference time: {result['inference_time_ms']:.0f}ms")
        print(f"   Output: {args.output}")

        # Print gate info if requested
        if args.gate_info and "gate_info" in result:
            print(f"\n🔍 Gate Analysis:")
            for entry in result["gate_info"]:
                if entry["activated"]:
                    print(
                        f"   Token {entry['token_idx']}: "
                        f"gate={entry['gate_value']:.3f} "
                        f"← matched '{entry.get('best_match_word', '?')}' "
                        f"(sim={entry.get('best_match_sim', 0):.3f})"
                    )


if __name__ == "__main__":
    main()
