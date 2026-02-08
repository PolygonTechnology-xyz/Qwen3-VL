#!/usr/bin/env python3
"""
Automated OCR dataset download + convert + finetune pipeline.
Usage: python scripts/train_ocr.py --model qwen3-vl-4b [other training args]
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qwenvl.data.download_and_convert import download_and_convert_ocr_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Download OCR dataset and finetune Qwen VL model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert dataset only (no training)
  python scripts/train_ocr.py --convert-only

  # Auto-convert dataset and train with Qwen3-VL-4B
  python scripts/train_ocr.py \\
    --model qwen3-vl-4b \\
    --max-samples 10000 \\
    --batch-size 8

  # Limit samples for quick testing
  python scripts/train_ocr.py \\
    --model qwen3-vl-4b \\
    --max-samples 100 \\
    --epochs 1
        """
    )
    
    # Dataset conversion arguments
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert dataset, don't train"
    )
    parser.add_argument(
        "--hf-dataset",
        default="kavinh07/nid-synth-200k-ocr",
        help="HuggingFace dataset ID (default: kavinh07/nid-synth-200k-ocr)"
    )
    parser.add_argument(
        "--output-dir",
        default="dataset",
        help="Output directory for converted dataset"
    )
    parser.add_argument(
        "--output-name",
        default="ocr_dataset",
        help="Output name prefix"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples (useful for testing)"
    )
    
    # Training arguments (will be passed to train_qwen.py)
    parser.add_argument(
        "--model",
        choices=["qwen3-vl-4b", "qwen2.5-vl-32b", "qwen2-vl-7b"],
        required=True,
        help="Model size"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Use LoRA fine-tuning"
    )
    
    args = parser.parse_args()
    
    # Step 1: Download and convert dataset
    print("\n" + "="*80)
    print("STEP 1: Download and Convert Dataset")
    print("="*80 + "\n")
    
    annotation_path, image_dir = download_and_convert_ocr_dataset(
        hf_dataset_id=args.hf_dataset,
        output_dir=args.output_dir,
        output_name=args.output_name,
        ocr_prompt="OCR the text. target language Bengali & English.",
        max_samples=args.max_samples,
        split="train"
    )
    
    if args.convert_only:
        print("\n✅ Dataset conversion complete!")
        print(f"   Annotations: {annotation_path}")
        print(f"   Images: {image_dir}")
        return 0
    
    # Step 2: Prepare training
    print("\n" + "="*80)
    print("STEP 2: Prepare Training Configuration")
    print("="*80 + "\n")
    
    # Model configurations
    model_configs = {
        "qwen3-vl-4b": {
            "model": "Qwen/Qwen3-VL-4B",
            "batch_size": args.batch_size,
            "grad_accum": 4,
        },
        "qwen2.5-vl-32b": {
            "model": "Qwen/Qwen2.5-VL-32B",
            "batch_size": args.batch_size,
            "grad_accum": 8,
        },
        "qwen2-vl-7b": {
            "model": "Qwen/Qwen2-VL-7B",
            "batch_size": args.batch_size,
            "grad_accum": 4,
        },
    }
    
    config = model_configs[args.model]
    
    # Build training command
    train_script = Path(__file__).parent.parent / "qwenvl" / "train" / "train_qwen.py"
    
    train_cmd = [
        "python", str(train_script),
        "--model_name_or_path", config["model"],
        "--data_path", image_dir,
        "--dataset_use", "ocr_dataset",
        "--output_dir", "output/ocr_finetune",
        "--num_train_epochs", str(args.epochs),
        "--per_device_train_batch_size", str(config["batch_size"]),
        "--gradient_accumulation_steps", str(config["grad_accum"]),
        "--learning_rate", str(args.lr),
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--max_pixels", "50176",
        "--min_pixels", "784",
        "--eval_strategy", "no",
        "--save_strategy", "steps",
        "--save_steps", "500",
        "--logging_steps", "10",
    ]
    
    if args.lora:
        train_cmd.extend([
            "--lora_enable", "True",
        ])
    
    print(f"Model: {args.model}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Gradient accumulation: {config['grad_accum']}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"LoRA: {args.lora}")
    print(f"\nDataset:")
    print(f"  Images: {image_dir}")
    print(f"  Annotations: {annotation_path}")
    
    # Step 3: Run training
    print("\n" + "="*80)
    print("STEP 3: Start Fine-tuning")
    print("="*80 + "\n")
    
    import subprocess
    result = subprocess.run(train_cmd)
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
