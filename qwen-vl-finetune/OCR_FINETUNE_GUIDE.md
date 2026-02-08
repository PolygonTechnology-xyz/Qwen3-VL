# OCR Dataset Fine-tuning Pipeline

## Overview

This pipeline automates the process of:
1. **Download** OCR dataset from HuggingFace (kavinh07/nid-synth-200k-ocr)
2. **Convert** to qwen-vl-finetune format (JSON + image folder)
3. **Fine-tune** Qwen VL models with the converted data

## Quick Start

### Option 1: One-Command Pipeline (Recommended)

```bash
# Download dataset + convert + train all at once
python scripts/train_ocr.py \
  --model qwen3-vl-4b \
  --batch-size 8 \
  --epochs 1

# With LoRA for faster training
python scripts/train_ocr.py \
  --model qwen3-vl-4b \
  --batch-size 8 \
  --lora
```

### Option 2: Convert Only (If you want to inspect/modify first)

```bash
# Just download and convert, no training
python -m qwenvl.data.download_and_convert \
  --mode convert \
  --output-dir dataset \
  --output-name ocr_dataset

# Or via standalone script
python scripts/train_ocr.py --convert-only
```

Then train manually:
```bash
# Register the dataset in qwenvl/data/__init__.py by uncommenting OCR_DATASET setup
# Then use it in training
python qwenvl/train/train_qwen.py \
  --model_name_or_path Qwen/Qwen3-VL-4B \
  --dataset_use "ocr_dataset" \
  --output_dir output/ocr_finetuned
```

## Parameters

### `train_ocr.py` Arguments

**Dataset Arguments:**
- `--hf-dataset`: HuggingFace dataset ID (default: `kavinh07/nid-synth-200k-ocr`)
- `--output-dir`: Where to save converted dataset (default: `dataset`)
- `--output-name`: Dataset name prefix (default: `ocr_dataset`)
- `--max-samples`: Limit samples (useful for testing, default: None = use all)
- `--convert-only`: Only convert, don't train

**Training Arguments:**
- `--model`: Model size - `qwen3-vl-4b`, `qwen2.5-vl-32b`, or `qwen2-vl-7b` (required)
- `--batch-size`: Training batch size (default: 4)
- `--epochs`: Number of epochs (default: 1)
- `--lr`: Learning rate (default: 1e-6)
- `--lora`: Enable LoRA fine-tuning

### Examples

```bash
# Quick test with 100 samples
python scripts/train_ocr.py \
  --model qwen3-vl-4b \
  --max-samples 100 \
  --epochs 1 \
  --batch-size 4

# Full training with LoRA
python scripts/train_ocr.py \
  --model qwen3-vl-4b \
  --batch-size 8 \
  --epochs 3 \
  --lr 2e-6 \
  --lora

# Use Qwen2.5-VL-32B (requires more VRAM)
python scripts/train_ocr.py \
  --model qwen2.5-vl-32b \
  --batch-size 4 \
  --epochs 1
```

## Dataset Format

The pipeline converts the HuggingFace dataset to this structure:

```
dataset/
├── ocr_dataset.json                    # Annotations
└── ocr_dataset_images/
    ├── ocr_dataset_000000.jpg
    ├── ocr_dataset_000001.jpg
    └── ...
```

### JSON Format

Each sample in `ocr_dataset.json`:
```json
{
  "image": "ocr_dataset_000000.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nOCR the text. target language Bengali & English."
    },
    {
      "from": "gpt",
      "value": "recognized text here"
    }
  ]
}
```

## Integration with Training

The dataset is automatically registered in `qwenvl/data/__init__.py` so you can use it:

```bash
# Via environment variable
python qwenvl/train/train_qwen.py \
  --model_name_or_path Qwen/Qwen3-VL-4B \
  --dataset_use "ocr_dataset" \
  --output_dir output/my_ocr_model

# With sampling (use 50% of dataset)
python qwenvl/train/train_qwen.py \
  --model_name_or_path Qwen/Qwen3-VL-4B \
  --dataset_use "ocr_dataset%50" \
  --output_dir output/my_ocr_model
```

## Advanced: Customizing the OCR Prompt

To modify the OCR prompt, edit `qwenvl/data/download_and_convert.py`:

```python
# Change the prompt here
download_and_convert_ocr_dataset(
    ...
    ocr_prompt="Your custom OCR prompt here",
    ...
)
```

Or pass it via command line:
```bash
python -m qwenvl.data.download_and_convert \
  --mode convert \
  --prompt "Extract all text from this image"
```

## Tips & Troubleshooting

### Memory Issues
- Use `--batch-size 4` or lower
- Enable LoRA with `--lora` to reduce VRAM
- Use `--max-samples` to test with fewer samples first

### Slow Conversion
- The model will download ~12GB of images on first run
- Conversion is done once and cached
- Check `dataset/ocr_dataset_images/` to verify download succeeded

### Custom Dataset Path
If you want to store the converted dataset elsewhere:
```bash
python -m qwenvl.data.download_and_convert \
  --output-dir /path/to/custom/location \
  --output-name my_ocr

# Then register in qwenvl/data/__init__.py
```

### Re-download/Reconvert
To force re-download and reconvert, delete the existing files:
```bash
rm dataset/ocr_dataset.json
rm -rf dataset/ocr_dataset_images/
```

## Output

After training completes:
- Checkpoints: `output/ocr_finetune/checkpoint-XXX/`
- Final model: `output/ocr_finetune/`

These can be loaded with:
```python
from transformers import AutoModel

model = AutoModel.from_pretrained("output/ocr_finetune", trust_remote_code=True)
```

## License

- Dataset: [kavinh07/nid-synth-200k-ocr](https://huggingface.co/datasets/kavinh07/nid-synth-200k-ocr)
- Code: Same as Qwen3-VL repository
