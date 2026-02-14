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
    ):
        """
        Args:
            eval_dataset: Evaluation dataset
            processor: Model processor for processing inputs
            output_dir: Directory to save outputs
            num_samples: Number of samples to generate per evaluation step
            max_new_tokens: Maximum new tokens to generate
            log_to_tensorboard: Whether to log results to tensorboard
        """
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.output_dir = output_dir
        self.num_samples = min(num_samples, len(eval_dataset) if eval_dataset else 5)
        self.max_new_tokens = max_new_tokens
        self.log_to_tensorboard = log_to_tensorboard
        self.generation_outputs = []
        self.trainer = None

        # Create outputs directory
        self.generation_dir = Path(output_dir) / "generation_outputs"
        self.generation_dir.mkdir(parents=True, exist_ok=True)

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """
        Called after each evaluation step to perform inference and log results.
        """
        if self.trainer is None or self.eval_dataset is None:
            return
        
        model = self.trainer.model

        device = model.device
        model.eval()

        generation_results = {
            "step": state.global_step,
            "generation_outputs": [],
        }

        # Sample a subset of evaluation data
        sample_indices = np.random.choice(len(self.eval_dataset), self.num_samples, replace=False)

        with torch.no_grad():
            for idx, sample_idx in enumerate(sample_indices):
                sample = self.eval_dataset[sample_idx]

                try:
                    # Prepare inputs
                    if isinstance(sample, dict):
                        # Extract inputs and images
                        input_ids = sample.get("input_ids", None)
                        pixel_values = sample.get("pixel_values", None)
                        image_grid_thw = sample.get("image_grid_thw", None)

                        # Prepare inputs for model
                        if pixel_values is not None:
                            if isinstance(pixel_values, list):
                                pixel_values = torch.stack(pixel_values)
                            pixel_values = pixel_values.to(device)

                        if input_ids is not None:
                            if isinstance(input_ids, list):
                                input_ids = torch.tensor(input_ids).unsqueeze(0)
                            elif len(input_ids.shape) == 1:
                                input_ids = input_ids.unsqueeze(0)
                            input_ids = input_ids.to(device)

                        # Prepare model inputs
                        model_inputs = {"input_ids": input_ids}
                        if pixel_values is not None:
                            if len(pixel_values.shape) == 3:
                                pixel_values = pixel_values.unsqueeze(0)
                            model_inputs["pixel_values"] = pixel_values

                        if image_grid_thw is not None:
                            if not isinstance(image_grid_thw, torch.Tensor):
                                image_grid_thw = torch.tensor(image_grid_thw)
                            model_inputs["image_grid_thw"] = image_grid_thw.to(device)

                        # Generate output
                        output_ids = model.generate(
                            **model_inputs,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,
                        )

                        # Decode outputs
                        generated_text = self.processor.decode(
                            output_ids[0][input_ids.shape[1] :],
                            skip_special_tokens=True,
                        )

                        # Prepare output record
                        output_record = {
                            "sample_index": int(sample_idx),
                            "generated_text": generated_text,
                            "input_ids_length": int(input_ids.shape[1]),
                            "output_ids_length": int(output_ids.shape[1]),
                        }

                        # Add reference text if available
                        if "labels" in sample:
                            ref_text = self.processor.decode(
                                sample["labels"], skip_special_tokens=True
                            )
                            output_record["reference_text"] = ref_text

                        generation_results["generation_outputs"].append(output_record)

                except Exception as e:
                    print(f"Error processing sample {sample_idx}: {e}")
                    generation_results["generation_outputs"].append({
                        "sample_index": int(sample_idx),
                        "error": str(e),
                    })

        # Save outputs to JSON file
        output_file = (
            self.generation_dir / f"generation_step_{state.global_step}.json"
        )
        with open(output_file, "w") as f:
            json.dump(generation_results, f, indent=2)

        print(f"Generation outputs saved to {output_file}")

        # Log to tensorboard if available
        if self.log_to_tensorboard and hasattr(model, "config"):
            try:
                # Try to get tensorboard writer from trainer callbacks
                import json
                from torch.utils.tensorboard import SummaryWriter

                # Log generation metrics
                writer = SummaryWriter(log_dir=Path(args.output_dir) / "runs")
                num_successful = len(
                    [o for o in generation_results["generation_outputs"] if "error" not in o]
                )
                writer.add_scalar(
                    "generation/successful_samples",
                    num_successful,
                    state.global_step,
                )
                writer.add_text(
                    "generation/outputs_summary",
                    json.dumps(generation_results, indent=2),
                    state.global_step,
                )
                writer.flush()
            except Exception as e:
                print(f"Could not log to tensorboard: {e}")

        model.train()
