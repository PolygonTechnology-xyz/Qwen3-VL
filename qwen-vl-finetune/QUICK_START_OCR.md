# Quick Reference: OCR Fine-tuning Setup

## 1️⃣ Prerequisites

```bash
# Ensure you have required dependencies
pip install huggingface_hub transformers datasets pillow tqdm

# (Should already be installed, but verify)
pip install -r requirements.txt
```

## 2️⃣ Download Dataset + Start Training

Choose **ONE** method:

### **Method A: Bash Script (Simplest)**
```bash
cd /mnt/data/github/Qwen3-VL/qwen-vl-finetune

# Download + convert + train (10k samples, 1 epoch)
bash scripts/train_ocr.sh

# Or with custom parameters
bash scripts/train_ocr.sh 50000 3    # 50k samples, 3 epochs
```

### **Method B: Python Script (More Control)**
```bash
cd /mnt/data/github/Qwen3-VL/qwen-vl-finetune

# Quick test (100 samples, Qwen3-VL-4B)
python scripts/train_ocr.py \
  --model qwen3-vl-4b \
  --max-samples 100 \
  --epochs 1

# Full training with LoRA
python scripts/train_ocr.py \
  --model qwen3-vl-4b \
  --batch-size 8 \
  --epochs 3 \
  --lr 2e-6 \
  --lora
```

### **Method C: Manual Steps**
```bash
cd /mnt/data/github/Qwen3-VL/qwen-vl-finetune

# Step 1: Download and convert dataset
python -m qwenvl.data.download_and_convert \
  --mode convert \
  --max-samples 10000 \
  --output-dir dataset \
  --output-name ocr_dataset

# Step 2: Start training
python qwenvl/train/train_qwen.py \
  --model_name_or_path Qwen/Qwen3-VL-4B \
  --dataset_use "ocr_dataset" \
  --data_path "dataset/ocr_dataset_images" \
  --output_dir "output/ocr_finetuned" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-6 \
  --warmup_ratio 0.03 \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500
```

## 📁 What Gets Created

```
dataset/
├── ocr_dataset.json              ← Annotations
└── ocr_dataset_images/
    ├── ocr_dataset_000000.jpg
    ├── ocr_dataset_000001.jpg
    └── ... (10-200k images)

output/
└── ocr_finetuned/               ← Fine-tuned model
    ├── checkpoint-500/
    ├── checkpoint-1000/
    └── adapter_config.json       ← (if using LoRA)
```

## 🎛️ Key Parameters

| Param | Default | Notes |
|-------|---------|-------|
| `--model` | - | `qwen3-vl-4b` (recommended), `qwen2.5-vl-32b`, `qwen2-vl-7b` |
| `--max-samples` | All | Use `100` or `1000` for quick tests |
| `--epochs` | 1 | Increase for better accuracy |
| `--batch-size` | 4 | Reduce if OOM errors |
| `--lr` | 1e-6 | Learning rate (1e-6 to 2e-6) |
| `--lora` | False | Enable for faster training with less VRAM |

## ⏱️ Timing

| Step | Duration | Notes |
|------|----------|-------|
| Download | 10-20 min | First time only, ~12GB |
| Convert | 10-30 min | Depends on sample count |
| Training (1 epoch) | 2-4 hours | Varies by GPU and model size |

## 💾 Storage Requirements

| Component | Size |
|-----------|------|
| Downloaded dataset images | ~12 GB |
| Converted JSON format | ~1-2 GB |
| Model checkpoints | ~7-50 GB (depending on model) |
| **Total** | ~20-65 GB |

## 🚀 Recommended First Run

```bash
# Test everything works with 100 samples, 1 epoch
python scripts/train_ocr.py \
  --model qwen3-vl-4b \
  --max-samples 100 \
  --epochs 1 \
  --batch-size 4

# Check output/ocr_finetuned/
```

Once working, increase `--max-samples` and `--epochs`:
```bash
# Full training
python scripts/train_ocr.py \
  --model qwen3-vl-4b \
  --max-samples 50000 \
  --epochs 3
```

## 🔧 Troubleshooting

**"ModuleNotFoundError: datasets"**
```bash
pip install datasets
```

**"CUDA out of memory"**
```bash
# Reduce batch size
--batch-size 4

# Or enable LoRA
--lora
```

**Want to change OCR prompt?**
Edit the prompt in:
- `scripts/train_ocr.py` (line 83)
- Or pass custom prompt to download script

**Dataset already partially downloaded?**
```bash
# Auto-resumes from where it left off
# Just run the command again
```

## 📖 Full Documentation

See [OCR_FINETUNE_GUIDE.md](OCR_FINETUNE_GUIDE.md) for advanced options and details.
