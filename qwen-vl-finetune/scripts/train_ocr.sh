#!/usr/bin/env bash
# Quick start: Download OCR dataset and fine-tune Qwen3-VL-4B
# Usage: bash scripts/train_ocr.sh [max_samples] [epochs]

set -e

MAX_SAMPLES=${1:-10000}  # Default: 10k samples (adjust as needed)
EPOCHS=${2:-1}           # Default: 1 epoch

echo "=========================================="
echo "🚀 OCR Fine-tuning Pipeline"
echo "=========================================="
echo "Max samples: $MAX_SAMPLES"
echo "Epochs: $EPOCHS"
echo ""

# Step 1: Download and convert dataset
echo "STEP 1: Downloading and converting dataset..."
python -m qwenvl.data.download_and_convert \
  --mode convert \
  --max-samples "$MAX_SAMPLES"

echo ""
echo "STEP 2: Starting fine-tuning..."
echo ""

# Step 2: Train
python qwenvl/train/train_qwen.py \
  --model_name_or_path Qwen/Qwen3-VL-4B \
  --dataset_use "ocr_dataset" \
  --data_path "dataset/ocr_dataset_images" \
  --output_dir "output/ocr_finetuned" \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-6 \
  --warmup_ratio 0.03 \
  --max_pixels 50176 \
  --min_pixels 784 \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --logging_steps 10

echo ""
echo "✅ Fine-tuning complete!"
echo "Model saved to: output/ocr_finetuned"
