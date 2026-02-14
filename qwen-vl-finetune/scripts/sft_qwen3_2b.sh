#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Distributed training configuration
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
# HuggingFace Hub configuration
HUB_TOKEN=${HUB_TOKEN:-""}  # Set your HuggingFace token here
HUB_MODEL_ID=${HUB_MODEL_ID:-""}  # Set your model repository name here

echo "HuggingFace Token: ${HUB_TOKEN}"
echo "HuggingFace Model ID: ${HUB_MODEL_ID}"

# DeepSpeed configuration
deepspeed=./scripts/zero2.json

# Model configuration
llm=Qwen/Qwen3-VL-2B-Instruct  # Using Qwen3-VL-4B model

# Training hyperparameters
lr=1e-5
batch_size=4
grad_accum_steps=4

# Dataset configuration
# Use local shards if available, otherwise download from HuggingFace
LOCAL_SHARDS_DIR=${LOCAL_SHARDS_DIR:-"/workspace/shards"}
DOWNLOAD_SHARDS=${DOWNLOAD_SHARDS:-false}  # Set to true to force download from HuggingFace
MAX_SAMPLES=${MAX_SAMPLES:-""}  # Leave empty for all samples, or set specific number
DATA_OUTPUT_DIR=dataset
DATA_OUTPUT_NAME=ocr_dataset

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Step 1: Download/convert OCR dataset
echo "="*80
echo "Step 1: Preparing OCR dataset..."
echo "="*80

# Check if local shards exist (unless download is forced)
if [ "$DOWNLOAD_SHARDS" = false ] && [ -d "$LOCAL_SHARDS_DIR" ] && [ "$(ls -A "$LOCAL_SHARDS_DIR"/*.tar 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "📂 Found local shards at: $LOCAL_SHARDS_DIR"
    SHARDS_COUNT=$(ls -1 "$LOCAL_SHARDS_DIR"/*.tar 2>/dev/null | wc -l)
    echo "📦 Converting $SHARDS_COUNT shards to dataset format..."
    
    python -m qwenvl.data.download_and_convert \
        --mode convert \
        --output-dir ${DATA_OUTPUT_DIR} \
        --output-name ${DATA_OUTPUT_NAME} \
        --local-shards-dir "${LOCAL_SHARDS_DIR}" \
        ${MAX_SAMPLES:+--max-samples $MAX_SAMPLES}
else
    if [ "$DOWNLOAD_SHARDS" = true ]; then
        echo "⬇️  Downloading from HuggingFace (DOWNLOAD_SHARDS=true)..."
    else
        echo "⚠️  Local shards not found at: $LOCAL_SHARDS_DIR"
        echo "⬇️  Downloading from HuggingFace instead..."
    fi
    
    python -m qwenvl.data.download_and_convert \
        --mode convert \
        --output-dir ${DATA_OUTPUT_DIR} \
        --output-name ${DATA_OUTPUT_NAME} \
        ${MAX_SAMPLES:+--max-samples $MAX_SAMPLES}
fi

if [ $? -ne 0 ]; then
    echo "❌ Dataset preparation failed!"
    exit 1
fi

echo "✅ Dataset ready at ${DATA_OUTPUT_DIR}/${DATA_OUTPUT_NAME}_images/"
echo ""

# Step 2: Prepare training configuration
datasets=${DATA_OUTPUT_NAME}
data_path="${DATA_OUTPUT_DIR}/${DATA_OUTPUT_NAME}_images"

# Output configuration
run_name="qwen3vl-ocr"
output_dir=./output

# Step 3: Launch training
echo "="*80
echo "Step 2: Starting fine-tuning..."
echo "="*80
echo "Model: ${llm}"
echo "Dataset: ${datasets}"
echo "Images: ${data_path}"
echo "Learning rate: ${lr}"
echo "Batch size: ${batch_size}"
echo "GPUs: ${NPROC_PER_NODE}"
echo "="*80
echo ""

# Training arguments
args="
    --hub_token ${HUB_TOKEN} \
    --hub_model_id ${HUB_MODEL_ID} \
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --lora_enable True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --eval_on_start True \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to tensorboard"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo "Model saved to: ${output_dir}"
else
    echo ""
    echo "❌ Training failed with exit code ${exit_code}"
fi

exit $exit_code
