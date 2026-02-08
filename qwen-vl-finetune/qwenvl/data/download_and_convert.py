"""
Download dataset from HuggingFace and convert to qwen-vl-finetune format.
Stores images in a folder and creates JSON annotations for training.
"""

import os
import json
import tarfile
import io
import math
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from PIL import Image

try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets: pip install datasets")
    load_dataset = None


def download_and_convert_ocr_dataset(
    hf_dataset_id: str = "kavinh07/nid-synth-200k-ocr",
    output_dir: str = "dataset",
    output_name: str = "ocr_dataset",
    ocr_prompt: str = "OCR the text. target language Bengali & English.",
    max_samples: Optional[int] = None,
    split: str = "train"
):
    """
    Download OCR dataset from HuggingFace and convert to qwen-vl-finetune format.
    
    Args:
        hf_dataset_id: HuggingFace dataset identifier
        output_dir: Directory to save converted dataset
        output_name: Name for the output files (e.g., 'ocr_dataset' -> 'ocr_dataset.json', 'ocr_dataset_images/')
        ocr_prompt: Prompt template for OCR task
        max_samples: Limit number of samples (None = use all)
        split: Dataset split to download (default: "train")
    
    Returns:
        Tuple of (annotation_path, image_dir)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_dir = os.path.join(output_dir, f"{output_name}_images")
    os.makedirs(image_dir, exist_ok=True)
    
    annotation_path = os.path.join(output_dir, f"{output_name}.json")
    
    # Check if already converted
    if os.path.exists(annotation_path):
        existing_count = len(json.load(open(annotation_path, 'r')))
        print(f"✓ Found existing dataset with {existing_count} samples at {annotation_path}")
        return annotation_path, image_dir
    
    print(f"📥 Downloading dataset from HuggingFace: {hf_dataset_id}...")
    if load_dataset is None:
        raise ImportError("Please install datasets: pip install datasets")
    
    dataset = load_dataset(hf_dataset_id, split=split)
    
    total_samples = len(dataset)
    if max_samples:
        total_samples = min(total_samples, max_samples)
        dataset = dataset.select(range(total_samples))
    
    print(f"📊 Converting {total_samples} samples...")
    
    annotations = []
    
    for idx, sample in enumerate(tqdm(dataset, total=total_samples, desc="Converting")):
        # Extract image and text
        # The dataset format from kavinh07/nid-synth-200k-ocr has 'image' and 'text' columns
        image = sample.get("image")
        text = sample.get("text", sample.get("label", ""))
        
        if image is None or not text:
            continue
        
        # Convert PIL Image to RGB if needed
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
        else:
            image = Image.open(image).convert("RGB")
        
        # Save image
        image_filename = f"{output_name}_{idx:06d}.jpg"
        image_path = os.path.join(image_dir, image_filename)
        image.save(image_path, "JPEG")
        
        # Create annotation entry
        annotation = {
            "image": image_filename,  # Relative path, will use data_path for full path
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{ocr_prompt}"
                },
                {
                    "from": "gpt",
                    "value": str(text).strip()
                }
            ]
        }
        annotations.append(annotation)
    
    # Save annotations
    with open(annotation_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Converted {len(annotations)} samples")
    print(f"   Images saved to: {image_dir}")
    print(f"   Annotations saved to: {annotation_path}")
    
    return annotation_path, image_dir


def download_and_create_shards(
    hf_dataset_id: str = "kavinh07/nid-synth-200k-ocr",
    output_dir: str = "shards",
    samples_per_shard: int = 2000,
    max_samples: Optional[int] = None,
    split: str = "train"
):
    """
    Alternative: Download dataset and create tar shards (for legacy support).
    
    Args:
        hf_dataset_id: HuggingFace dataset identifier
        output_dir: Directory to save shards
        samples_per_shard: Number of samples per shard
        max_samples: Limit number of samples
        split: Dataset split to download
    
    Returns:
        Path to shards directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if shards already exist
    existing_shards = sorted([f for f in os.listdir(output_dir) if f.endswith(".tar")])
    if existing_shards:
        print(f"✓ Found {len(existing_shards)} existing shards in {output_dir}")
        return output_dir
    
    print(f"📥 Downloading dataset from HuggingFace: {hf_dataset_id}...")
    if load_dataset is None:
        raise ImportError("Please install datasets: pip install datasets")
    
    dataset = load_dataset(hf_dataset_id, split=split)
    
    total_samples = len(dataset)
    if max_samples:
        total_samples = min(total_samples, max_samples)
        dataset = dataset.select(range(total_samples))
    
    num_shards = math.ceil(total_samples / samples_per_shard)
    
    print(f"📦 Creating {num_shards} shards from {total_samples} samples...")
    
    for shard_id in tqdm(range(num_shards), desc="Sharding"):
        start = shard_id * samples_per_shard
        end = min((shard_id + 1) * samples_per_shard, total_samples)
        
        shard_path = os.path.join(output_dir, f"shard-{shard_id:05d}.tar")
        
        shard_dataset = dataset.select(range(start, end))
        
        with tarfile.open(shard_path, "w") as tar:
            for sample_idx, sample in enumerate(shard_dataset):
                image = sample.get("image")
                text = sample.get("text", sample.get("label", ""))
                
                if image is None or not text:
                    continue
                
                # Convert to RGB
                if isinstance(image, Image.Image):
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                else:
                    image = Image.open(image).convert("RGB")
                
                # Save image to tar
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="JPEG")
                img_data = img_bytes.getvalue()
                
                sample_num = start + sample_idx
                
                # Add image to tar
                img_tarinfo = tarfile.TarInfo(name=f"sample_{sample_num:06d}.jpg")
                img_tarinfo.size = len(img_data)
                tar.addfile(tarinfo=img_tarinfo, fileobj=io.BytesIO(img_data))
                
                # Add text to tar
                text_bytes = str(text).strip().encode("utf-8")
                txt_tarinfo = tarfile.TarInfo(name=f"sample_{sample_num:06d}.txt")
                txt_tarinfo.size = len(text_bytes)
                tar.addfile(tarinfo=txt_tarinfo, fileobj=io.BytesIO(text_bytes))
    
    print(f"✅ Created {num_shards} shards in {output_dir}")
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and convert OCR dataset")
    parser.add_argument("--hf-dataset", default="kavinh07/nid-synth-200k-ocr", help="HuggingFace dataset ID")
    parser.add_argument("--output-dir", default="dataset", help="Output directory")
    parser.add_argument("--output-name", default="ocr_dataset", help="Output name prefix")
    parser.add_argument("--prompt", default="OCR the text. target language Bengali & English.", help="OCR prompt")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--mode", choices=["convert", "shards"], default="convert", help="Mode: convert to JSON or create shards")
    
    args = parser.parse_args()
    
    if args.mode == "convert":
        download_and_convert_ocr_dataset(
            hf_dataset_id=args.hf_dataset,
            output_dir=args.output_dir,
            output_name=args.output_name,
            ocr_prompt=args.prompt,
            max_samples=args.max_samples
        )
    else:
        download_and_create_shards(
            hf_dataset_id=args.hf_dataset,
            output_dir=args.output_dir,
            max_samples=args.max_samples
        )
