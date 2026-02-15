# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer, EarlyStoppingCallback
from qwenvl.train.callbacks import CustomGenerationCallback
from qwenvl.train.metrics import compute_metrics_batch

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

def preprocess_logits_for_metrics(logits, labels):
    """
    Convert raw logits to predicted token IDs before they reach compute_metrics.
    The Trainer passes raw logits by default; without this, compute_metrics
    receives floats instead of token IDs.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def create_compute_metrics_fn(processor):
    """
    Create a compute_metrics function for CER and WER evaluation.
    
    Args:
        processor: Model processor for decoding
    
    Returns:
        compute_metrics function compatible with Trainer
    """
    def compute_metrics(eval_preds):
        from transformers.trainer_utils import EvalPrediction
        
        predictions, labels = eval_preds
        
        # Decode predictions and labels
        decoded_preds = []
        decoded_labels = []
        
        for pred, label in zip(predictions, labels):
            # Handle variable length sequences
            if isinstance(pred, (list, tuple)):
                pred_ids = pred
            else:
                pred_ids = pred.tolist() if hasattr(pred, 'tolist') else pred
            
            if isinstance(label, (list, tuple)):
                label_ids = label
            else:
                label_ids = label.tolist() if hasattr(label, 'tolist') else label
            
            # Ensure pred_ids and label_ids are flat 1D lists
            if isinstance(pred_ids, list) and len(pred_ids) > 0 and isinstance(pred_ids[0], (list, tuple)):
                pred_ids = pred_ids[0]
            if isinstance(label_ids, list) and len(label_ids) > 0 and isinstance(label_ids[0], (list, tuple)):
                label_ids = label_ids[0]
            
            # Convert to integers (they should already be token IDs from preprocess_logits_for_metrics)
            pred_ids = [int(p) for p in pred_ids if int(p) != -100]
            
            # Replace -100 (ignored tokens) with pad token, filter out -100 from labels  
            label_ids = [int(l) for l in label_ids if int(l) != -100]
            
            # Filter out-of-range token IDs as a safety net
            vocab_size = len(processor.tokenizer)
            pred_ids = [p for p in pred_ids if 0 <= p < vocab_size]
            label_ids = [l for l in label_ids if 0 <= l < vocab_size]
            
            # Skip if empty
            if not pred_ids or not label_ids:
                continue
            
            try:
                decoded_pred = processor.tokenizer.decode(pred_ids, skip_special_tokens=True)
                decoded_label = processor.tokenizer.decode(label_ids, skip_special_tokens=True)
                
                decoded_preds.append(decoded_pred)
                decoded_labels.append(decoded_label)
            except Exception as e:
                rank0_print(f"Error decoding: {e}")
                continue
        
        # Compute CER and WER
        if decoded_preds and decoded_labels:
            metrics_result = compute_metrics_batch(decoded_labels, decoded_preds, normalize=True)
            return {
                "cer": metrics_result["cer"],
                "wer": metrics_result["wer"],
            }
        else:
            return {
                "cer": 0.0,
                "wer": 0.0,
            }
    
    return compute_metrics


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        if torch.distributed.get_rank() == 0:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()
    
    data_module = make_supervised_data_module(processor, data_args=data_args)

    callbacks = []
    callbacks.append(
        EarlyStoppingCallback(
            early_stopping_patience=5
        )
    )

    # if data_module.get("eval_dataset") is not None:
    #     generation_callback = CustomGenerationCallback(
    #         eval_dataset=data_module["eval_dataset"],
    #         model=model,
    #         processor=processor,
    #         output_dir=training_args.output_dir,
    #         log_to_tensorboard=True,
    #         dataset_json_path="dataset/ocr_dataset.json",
    #         dataset_images_dir="dataset/ocr_dataset_images",
    #     )
    #     callbacks.append(generation_callback)
    #     rank0_print("Generation callback enabled (loads fresh samples from OCR dataset for generation)")
    
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, callbacks=callbacks, **data_module
    )
    try:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            logging.info("checkpoint found, resume training")
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
    except Exception as e:
        logging.error(f"Training failed: {e}")

    finally:
        if training_args.hub_model_id and training_args.hub_token:        
            trainer.push_to_hub()
        else:
            logging.info("HuggingFace Hub credentials not provided, skipping push to hub.")

    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
