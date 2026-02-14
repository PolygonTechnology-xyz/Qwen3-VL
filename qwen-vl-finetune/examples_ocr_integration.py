"""
Example: Using the downloaded OCR dataset with your own training code.
Shows how to:
1. Auto-download and convert the dataset
2. Load it manually for custom training
3. Integrate with the qwen-vl-finetune framework
"""

import os
import json
from pathlib import Path


# ======================================================================
# Example 1: Auto-download + convert in your own training code
# ======================================================================

def example_auto_download_and_train():
    from qwenvl.data.download_and_convert import download_and_convert_ocr_dataset
    
    # Download and convert to qwen-vl format
    annotation_path, image_dir = download_and_convert_ocr_dataset(
        hf_dataset_id="kavinh07/nid-synth-200k-ocr",
        output_dir="dataset",
        output_name="ocr_dataset",
        ocr_prompt="OCR the text. target language Bengali & English.",
        max_samples=10000,  # Use None for all samples
        split="train"
    )
    
    print(f"Annotations: {annotation_path}")
    print(f"Images: {image_dir}")
    
    # Now use these paths in your training script
    return annotation_path, image_dir


# ======================================================================
# Example 2: Load converted dataset and inspect it
# ======================================================================

def example_load_and_inspect():
    import json
    
    # Load annotations
    with open("dataset/ocr_dataset.json", "r") as f:
        samples = json.load(f)
    
    print(f"Total samples: {len(samples)}")
    
    # Inspect first sample
    sample = samples[0]
    print(f"\nFirst sample:")
    print(f"  Image: {sample['image']}")
    print(f"  Human: {sample['conversations'][0]['value'][:50]}...")
    print(f"  GPT: {sample['conversations'][1]['value']}")
    
    # Count images
    image_dir = "dataset/ocr_dataset_images"
    image_count = len(os.listdir(image_dir))
    print(f"\nTotal images: {image_count}")


# ======================================================================
# Example 3: Use with qwen-vl-finetune framework
# ======================================================================

def example_finetune_integration():
    """
    After running download_and_convert_ocr_dataset(), use it with the framework:
    """
    
    # Option A: Via command line
    cmd = """
    python qwenvl/train/train_qwen.py \\
      --model_name_or_path Qwen/Qwen3-VL-4B \\
      --dataset_use "ocr_dataset" \\
      --data_path dataset/ocr_dataset_images \\
      --output_dir output/ocr_finetuned \\
      --num_train_epochs 1 \\
      --per_device_train_batch_size 8 \\
      --gradient_accumulation_steps 4 \\
      --learning_rate 1e-6
    """
    print("Command line:", cmd)
    
    # Option B: Via Python (if framework supports it)
    # from qwenvl.train.train_qwen import main
    # main([
    #     "--model_name_or_path", "Qwen/Qwen3-VL-4B",
    #     "--dataset_use", "ocr_dataset",
    #     "--data_path", "dataset/ocr_dataset_images",
    #     ...
    # ])


# ======================================================================
# Example 4: Create shards instead of JSON (if preferred)
# ======================================================================

def example_create_shards():
    from qwenvl.data.download_and_convert import download_and_create_shards
    
    # Download and create tar shards
    shards_dir = download_and_create_shards(
        hf_dataset_id="kavinh07/nid-synth-200k-ocr",
        output_dir="shards",
        samples_per_shard=2000,
        max_samples=10000
    )
    
    print(f"Shards created in: {shards_dir}")
    
    # List shards
    shard_files = sorted([f for f in os.listdir(shards_dir) if f.endswith(".tar")])
    print(f"Shard files: {shard_files}")


# ======================================================================
# Example 5: Custom prompt for different OCR tasks
# ======================================================================

def example_custom_prompts():
    from qwenvl.data.download_and_convert import download_and_convert_ocr_dataset
    
    # Different prompts for different subtasks
    prompts = {
        "basic": "OCR the text in this image.",
        "multilingual": "OCR the text. target language Bengali & English.",
        "handwriting": "Recognize the handwritten text in this image.",
        "scene_text": "Extract all text visible in this scene image.",
    }
    
    for task, prompt in prompts.items():
        annotation_path, image_dir = download_and_convert_ocr_dataset(
            hf_dataset_id="kavinh07/nid-synth-200k-ocr",
            output_dir=f"dataset/ocr_{task}",
            output_name=f"ocr_{task}",
            ocr_prompt=prompt,
            max_samples=1000
        )


# ======================================================================
# Example 6: Loading dataset for custom processing
# ======================================================================

def example_custom_dataset_class():
    """
    If you want to write your own dataset class:
    """
    import json
    from pathlib import Path
    from PIL import Image
    
    class OCRDataset:
        def __init__(self, annotation_path, image_dir):
            with open(annotation_path, "r") as f:
                self.samples = json.load(f)
            self.image_dir = image_dir
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            sample = self.samples[idx]
            
            # Load image
            image_path = Path(self.image_dir) / sample["image"]
            image = Image.open(image_path).convert("RGB")
            
            # Extract conversation
            human_text = sample["conversations"][0]["value"]
            gt_text = sample["conversations"][1]["value"]
            
            return {
                "image": image,
                "prompt": human_text,
                "ground_truth": gt_text
            }
    
    # Usage
    dataset = OCRDataset(
        "dataset/ocr_dataset.json",
        "dataset/ocr_dataset_images"
    )
    
    sample = dataset[0]
    print(f"Image shape: {sample['image'].size}")
    print(f"Prompt: {sample['prompt']}")
    print(f"Ground truth: {sample['ground_truth']}")


# ======================================================================
# Example 7: Batch processing with sampling
# ======================================================================

def example_sampling():
    from qwenvl.data.download_and_convert import download_and_convert_ocr_dataset
    
    # Small sample for quick testing
    download_and_convert_ocr_dataset(
        output_name="ocr_test",
        max_samples=100
    )
    
    # Medium sample for development
    download_and_convert_ocr_dataset(
        output_name="ocr_dev",
        max_samples=10000
    )
    
    # Full dataset for production
    download_and_convert_ocr_dataset(
        output_name="ocr_full",
        max_samples=None  # All samples
    )
    
    # Register all in qwenvl/data/__init__.py and use with:
    # python train_qwen.py --dataset_use "ocr_test,ocr_dev,ocr_full%10"
    # (uses test + dev + 10% of full)


if __name__ == "__main__":
    import sys
    
    examples = {
        "1": ("Auto-download and train", example_auto_download_and_train),
        "2": ("Load and inspect dataset", example_load_and_inspect),
        "3": ("Framework integration", example_finetune_integration),
        "4": ("Create shards", example_create_shards),
        "5": ("Custom prompts", example_custom_prompts),
        "6": ("Custom dataset class", example_custom_dataset_class),
        "7": ("Batch processing/sampling", example_sampling),
    }
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num in examples:
            print(f"\n{'='*70}")
            print(f"Running: {examples[example_num][0]}")
            print(f"{'='*70}\n")
            examples[example_num][1]()
        else:
            print(f"Example {example_num} not found")
    else:
        print("Available examples:")
        for num, (name, func) in examples.items():
            print(f"  {num}. {name}")
        print("\nUsage: python examples.py <1-7>")
