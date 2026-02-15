import os
import json
import torch
from typing import Optional, Dict, Any, List
from pathlib import Path

from transformers import TrainerCallback, TrainerState, TrainerControl
import numpy as np


class CustomGenerationCallback(TrainerCallback):
    """
    Custom callback that performs inference on evaluation data and logs to tensorboard at each evaluation step.
    Saves generation outputs for tensorboard visualization.
    """

    def __init__(
        self,
        eval_dataset,
        processor,
        model,
        output_dir: str,
        num_samples: int = 5,
        max_new_tokens: int = 100,
        log_to_tensorboard: bool = True,
        dataset_json_path: str = None,
        dataset_images_dir: str = None,
    ):
        """
        Args:
            eval_dataset: Evaluation dataset
            processor: Model processor for processing inputs
            output_dir: Directory to save outputs
            num_samples: Number of samples to generate per evaluation step
            max_new_tokens: Maximum new tokens to generate
            log_to_tensorboard: Whether to log results to tensorboard
            dataset_json_path: Path to original OCR dataset JSON for fresh processing
            dataset_images_dir: Directory containing OCR dataset images
        """
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.model = model
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.log_to_tensorboard = log_to_tensorboard
        self.generation_outputs = []
        self.trainer = None
        self.dataset_json_path = dataset_json_path
        self.dataset_images_dir = dataset_images_dir
        self.sample_indices = None  # Will be set on first evaluation for consistent sampling

        # Create outputs directory
        self.generation_dir = Path(output_dir) / "generation_outputs"
        self.generation_dir.mkdir(parents=True, exist_ok=True)

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """
        Called after each evaluation step to perform inference and log results.
        Uses eval_dataset passed during initialization.
        Uses fixed sample indices for consistent evaluation across steps.
        """
        # if self.trainer is None:
        #     return
        
        # Check if we have eval_dataset
        if self.eval_dataset is None:
            print(f"\n⚠️ No eval_dataset provided - skipping generation callback")
            return
        
        model = self.model
        device = model.device
        model.eval()
        
        print(f"\n{'='*60}")
        print(f"🔄 Running generation callback at step {state.global_step}...")
        print(f"📂 Using eval_dataset")
        print(f"{'='*60}\n")

        generation_results = {
            "step": state.global_step,
            "generation_outputs": [],
        }

        with torch.no_grad():
            try:
                # Get total dataset size
                total_samples = len(self.eval_dataset)
                print(f"📖 Eval dataset size: {total_samples}")
                
                # Initialize sample indices on first evaluation (constant for all steps)
                if self.sample_indices is None:
                    self.sample_indices = np.random.choice(total_samples, min(self.num_samples, total_samples), replace=False)
                    print(f"🔧 Initializing fixed sample indices for all evaluations: {self.sample_indices.tolist()}")
                
                sample_indices = self.sample_indices
                print(f"📊 Using fixed sample indices: {sample_indices.tolist()}\n")
                
                for idx, sample_idx in enumerate(sample_indices):
                    print(f"\n--- Processing sample {idx+1}/{len(sample_indices)} (dataset index {sample_idx}) ---")
                    
                    try:
                        # Get sample from eval_dataset (already tokenized)
                        sample = self.eval_dataset[int(sample_idx)]
                        
                        # Dataset is pre-tokenized with: input_ids, labels, pixel_values, image_grid_thw, position_ids
                        input_ids = sample['input_ids']
                        labels = sample['labels']
                        
                        # Convert to tensors if needed
                        if not isinstance(input_ids, torch.Tensor):
                            input_ids = torch.tensor(input_ids)
                        if not isinstance(labels, torch.Tensor):
                            labels = torch.tensor(labels)
                        
                        # Extract prompt portion: where labels == -100 (masked tokens are the prompt)
                        prompt_mask = (labels == -100)
                        prompt_ids = input_ids[prompt_mask]
                        
                        # Extract reference: non -100 label tokens
                        reference_ids = labels[labels != -100]
                        reference_text = self.processor.tokenizer.decode(
                            reference_ids, skip_special_tokens=True
                        )
                        
                        # Decode prompt for display
                        prompt_text = self.processor.tokenizer.decode(
                            prompt_ids, skip_special_tokens=True
                        )
                        
                        print(f"  💬 Prompt: {prompt_text[:80]}...")
                        print(f"  ✅ Reference: {reference_text[:80]}...")
                        
                        # Build model inputs from the prompt portion
                        prompt_ids_batched = prompt_ids.unsqueeze(0).to(device)
                        
                        # Prepare generate inputs
                        generate_inputs = {
                            'input_ids': prompt_ids_batched,
                        }
                        
                        # Add pixel_values and image_grid_thw if present
                        if 'pixel_values' in sample:
                            pv = sample['pixel_values']
                            if not isinstance(pv, torch.Tensor):
                                pv = torch.tensor(pv)
                            generate_inputs['pixel_values'] = pv.unsqueeze(0).to(device) if pv.dim() == 1 else pv.to(device)
                        
                        if 'image_grid_thw' in sample:
                            igt = sample['image_grid_thw']
                            if not isinstance(igt, torch.Tensor):
                                igt = torch.tensor(igt)
                            generate_inputs['image_grid_thw'] = igt.to(device) if igt.dim() == 2 else igt.unsqueeze(0).to(device)
                        
                        print(f"  🔧 Prompt input_ids shape: {prompt_ids_batched.shape}")
                        
                        # Generate
                        print(f"  🤖 Generating...")
                        generated_ids = model.generate(
                            **generate_inputs,
                            max_new_tokens=self.max_new_tokens
                        )
                        
                        # Trim to only new tokens
                        input_length = prompt_ids_batched.shape[1]
                        generated_ids_trimmed = [out_ids[input_length:] for out_ids in generated_ids]
                        
                        # Decode
                        output_text = self.processor.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        )
                        
                        generated_text = output_text[0] if output_text else ""
                        
                        print(f"  ✨ Generated: {generated_text[:80]}...")
                        
                        # Save results
                        output_record = {
                            "sample_index": int(sample_idx),
                            "prompt": prompt_text,
                            "generated_text": generated_text,
                            "reference_text": reference_text,
                        }
                        
                        generation_results["generation_outputs"].append(output_record)
                        print(f"✅ Successfully generated for sample {idx+1}")
                    
                    except Exception as e:
                        print(f"❌ Error processing sample {sample_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                        generation_results["generation_outputs"].append({
                            "sample_index": int(sample_idx),
                            "error": str(e),
                        })
            
            except Exception as e:
                print(f"❌ Fatal error in generation callback: {e}")
                import traceback
                traceback.print_exc()
                # Restore attention implementation before returning
                if original_attn_implementation is not None and hasattr(model.config, '_attn_implementation'):
                    model.config._attn_implementation = original_attn_implementation
                model.train()
                return
        
        # Print summary
        num_successful = len([o for o in generation_results["generation_outputs"] if "error" not in o])
        num_failed = len([o for o in generation_results["generation_outputs"] if "error" in o])
        print(f"\n📊 Generation summary: {num_successful} successful, {num_failed} failed out of {len(sample_indices)} samples")

        # Save outputs to JSON file
        output_file = (
            self.generation_dir / f"generation_step_{state.global_step}.json"
        )
        # with open(output_file, "w") as f:
        #     json.dump(generation_results, f, indent=2)

        # print(f"📝 Generation outputs saved to {output_file}")

        # Log to tensorboard
        if self.log_to_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                
                # Use a separate subdirectory to avoid conflicting with the trainer's TensorBoard writer
                if hasattr(args, 'logging_dir') and args.logging_dir:
                    log_dir = Path(args.logging_dir) / "generation"
                else:
                    log_dir = Path(args.output_dir) / "runs" / "generation"
                
                print(f"📊 Using TensorBoard log directory: {log_dir}")
                
                # Create a SummaryWriter in the generation subdirectory
                tb_writer = SummaryWriter(log_dir=str(log_dir))
                
                # Count successful generations
                num_successful = len(
                    [o for o in generation_results["generation_outputs"] if "error" not in o]
                )
                
                print(f"📈 Logging generation samples to TensorBoard...")
                
                # Log scalar: number of successful samples
                tb_writer.add_scalar(
                    "generation/successful_samples",
                    num_successful,
                    state.global_step,
                )
                print(f"  ✓ Logged scalar: generation/successful_samples = {num_successful}")
                
                # Consolidate all samples into a single text block
                consolidated_lines = [
                    f"# Generation Results - Step {state.global_step}",
                    f"**Successful:** {num_successful} / {len(sample_indices)}",
                    "",
                ]
                
                for idx, output in enumerate(generation_results["generation_outputs"]):
                    if "error" not in output:
                        consolidated_lines.append(f"## Sample {idx + 1} - Index {output.get('sample_index', idx)}")
                        consolidated_lines.append("")
                        consolidated_lines.append("**Prompt:**")
                        consolidated_lines.append(f"> {output.get('prompt', 'N/A')}")
                        consolidated_lines.append("")
                        consolidated_lines.append("**Generated Output:**")
                        consolidated_lines.append(f"> {output.get('generated_text', 'N/A')}")
                        consolidated_lines.append("")
                        consolidated_lines.append("**Reference (Ground Truth):**")
                        consolidated_lines.append(f"> {output.get('reference_text', 'N/A')}")
                        consolidated_lines.append("")
                    else:
                        consolidated_lines.append(f"## Sample {idx + 1} - Index {output.get('sample_index', idx)}")
                        consolidated_lines.append(f"❌ Error: {output.get('error', 'Unknown error')}")
                        consolidated_lines.append("")
                
                consolidated_text = "\n".join(consolidated_lines)
                tb_writer.add_text("generation/all_samples", consolidated_text, state.global_step)
                print(f"  ✓ Logged consolidated text: generation/all_samples")
                
                # Force flush to disk
                tb_writer.flush()
                tb_writer.close()
                
                print(f"✅ Successfully logged to TensorBoard at step {state.global_step}")
                print(f"   View at: tensorboard --logdir={log_dir}")
                    
            except Exception as e:
                print(f"❌ TensorBoard logging error: {e}")
                import traceback
                traceback.print_exc()
        
        model.train()
        print(f"{'='*60}\n")