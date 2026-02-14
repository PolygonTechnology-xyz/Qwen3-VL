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
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.log_to_tensorboard = log_to_tensorboard
        self.generation_outputs = []
        self.trainer = None
        self.dataset_json_path = dataset_json_path
        self.dataset_images_dir = dataset_images_dir

        # Create outputs directory
        self.generation_dir = Path(output_dir) / "generation_outputs"
        self.generation_dir.mkdir(parents=True, exist_ok=True)

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """
        Called after each evaluation step to perform inference and log results.
        Loads fresh samples from OCR dataset JSON to avoid packed format issues.
        """
        if self.trainer is None:
            return
        
        # Check if we have dataset paths for fresh generation
        if not self.dataset_json_path or not self.dataset_images_dir:
            print(f"\n⚠️ No dataset_json_path or dataset_images_dir provided - skipping generation callback")
            return
        
        model = self.trainer.model
        device = model.device
        model.eval()
        
        # Store original attention implementation and switch to eager for generation
        # Flash attention has issues with generation in our setup
        original_attn_implementation = None
        if hasattr(model.config, '_attn_implementation'):
            original_attn_implementation = model.config._attn_implementation
            model.config._attn_implementation = 'eager'
            print(f"⚙️  Temporarily switching attention from '{original_attn_implementation}' to 'eager' for generation")
        
        print(f"\n{'='*60}")
        print(f"🔄 Running generation callback at step {state.global_step}...")
        print(f"📂 Loading fresh samples from: {self.dataset_json_path}")
        print(f"{'='*60}\n")

        generation_results = {
            "step": state.global_step,
            "generation_outputs": [],
        }

        with torch.no_grad():
            try:
                # Load raw OCR dataset
                print(f"📖 Loading OCR dataset...")
                with open(self.dataset_json_path, 'r') as f:
                    raw_dataset = json.load(f)
                
                # Sample random indices
                total_samples = len(raw_dataset)
                sample_indices = np.random.choice(total_samples, min(self.num_samples, total_samples), replace=False)
                print(f"📊 Total dataset size: {total_samples}")
                print(f"📊 Selected sample indices: {sample_indices.tolist()}\n")
                
                for idx, sample_idx in enumerate(sample_indices):
                    print(f"\n--- Processing sample {idx+1}/{len(sample_indices)} (dataset index {sample_idx}) ---")
                    
                    try:
                        # Get raw sample
                        raw_sample = raw_dataset[sample_idx]
                        image_filename = raw_sample["image"]
                        conversations = raw_sample["conversations"]
                        
                        # Extract user prompt and reference text
                        user_prompt = conversations[0]["value"]  # Human question
                        reference_text = conversations[1]["value"]  # GPT answer
                        
                        # Build image path
                        image_path = os.path.join(self.dataset_images_dir, image_filename)
                        
                        if not os.path.exists(image_path):
                            raise FileNotFoundError(f"Image not found: {image_path}")
                        
                        print(f"  📷 Image: {image_filename}")
                        print(f"  💬 Prompt: {user_prompt[:80]}...")
                        print(f"  ✅ Reference: {reference_text[:80]}...")
                        
                        # Process using official Qwen3VL pattern
                        from PIL import Image
                        image = Image.open(image_path)
                        
                        # Create messages in chat format
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": image},
                                    {"type": "text", "text": user_prompt.replace("<image>\n", "")},
                                ],
                            }
                        ]
                        
                        # Process with processor (official pattern)
                        text = self.processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        inputs = self.processor(
                            text=[text],
                            images=[image],
                            padding=True,
                            return_tensors="pt"
                        )
                        inputs = inputs.to(device)
                        
                        print(f"  🔧 Processed inputs - input_ids shape: {inputs['input_ids'].shape}")
                        
                        # Generate
                        print(f"  🤖 Generating...")
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens
                        )
                        
                        # Trim to only new tokens
                        input_length = inputs['input_ids'].shape[1]
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
                            "image": image_filename,
                            "prompt": user_prompt,
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
        with open(output_file, "w") as f:
            json.dump(generation_results, f, indent=2)

        print(f"📝 Generation outputs saved to {output_file}")

        # Log to tensorboard
        if self.log_to_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                
                # Determine the log directory
                if hasattr(args, 'logging_dir') and args.logging_dir:
                    log_dir = Path(args.logging_dir)
                else:
                    log_dir = Path(args.output_dir) / "runs"
                
                print(f"📊 Using TensorBoard log directory: {log_dir}")
                
                # Create a SummaryWriter
                tb_writer = SummaryWriter(log_dir=str(log_dir))
                
                # Count successful generations
                num_successful = len(
                    [o for o in generation_results["generation_outputs"] if "error" not in o]
                )
                
                print(f"📈 Logging {num_successful} generation samples to TensorBoard...")
                
                # Log scalar: number of successful samples
                tb_writer.add_scalar(
                    "generation/successful_samples",
                    num_successful,
                    state.global_step,
                )
                print(f"  ✓ Logged scalar: generation/successful_samples = {num_successful}")
                
                # Log each generated output as text
                for idx, output in enumerate(generation_results["generation_outputs"]):
                    if "error" not in output:
                        # Create a formatted markdown text
                        text_lines = [
                            f"### Sample {output.get('sample_index', idx)} (Step {state.global_step})",
                            "",
                            f"**Image:** `{output.get('image', 'N/A')}`",
                            "",
                            "**Prompt:**",
                            f"> {output.get('prompt', 'N/A')[:100]}...",
                            "",
                            "**Generated Output:**",
                            f"> {output.get('generated_text', 'N/A')}",
                            "",
                            "**Reference (Ground Truth):**",
                            f"> {output.get('reference_text', 'N/A')}",
                            "",
                        ]
                        
                        text_content = "\n".join(text_lines)
                        
                        tag = f"generation_samples/sample_{idx}"
                        tb_writer.add_text(tag, text_content, state.global_step)
                        print(f"  ✓ Logged sample: {tag}")
                
                # Log full summary as formatted text
                summary_lines = [
                    "### Generation Summary", 
                    "", 
                    f"**Step:** {state.global_step}", 
                    f"**Successful:** {num_successful}",
                    ""
                ]
                for idx, output in enumerate(generation_results["generation_outputs"]):
                    if "error" not in output:
                        summary_lines.append(f"**Sample {idx} ({output.get('image', 'N/A')}):**")
                        summary_lines.append(f"- Generated: {output.get('generated_text', 'N/A')[:80]}...")
                        summary_lines.append(f"- Reference:  {output.get('reference_text', 'N/A')[:80]}...")
                        summary_lines.append("")
                
                summary_text = "\n".join(summary_lines)
                tb_writer.add_text("generation_summary/all_samples", summary_text, state.global_step)
                print(f"  ✓ Logged text: generation_summary/all_samples")
                
                # Force flush to disk
                tb_writer.flush()
                tb_writer.close()
                
                print(f"✅ Successfully logged to TensorBoard at step {state.global_step}")
                print(f"   View at: tensorboard --logdir={log_dir}")
                    
            except Exception as e:
                print(f"❌ TensorBoard logging error: {e}")
                import traceback
                traceback.print_exc()
        
        # Restore original attention implementation
        if original_attn_implementation is not None and hasattr(model.config, '_attn_implementation'):
            model.config._attn_implementation = original_attn_implementation
            print(f"⚙️  Restored attention implementation to '{original_attn_implementation}'")
        
        model.train()
        print(f"{'='*60}\n")

