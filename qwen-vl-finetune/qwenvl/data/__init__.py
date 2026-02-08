import re
import os
from pathlib import Path

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

dataset = {
    "annotation_path": "/mnt/data/github/Qwen3-VL/qwen-vl-finetune/dataset/dataset.json",
    "data_path": "/mnt/data/github/Qwen3-VL/qwen-vl-finetune/dataset",
}


def _setup_ocr_dataset_if_needed(
    output_dir="dataset",
    output_name="ocr_dataset",
    max_samples=None
):
    """
    Auto-download and convert OCR dataset on first use.
    """
    import os
    from pathlib import Path
    
    annotation_path = os.path.join(output_dir, f"{output_name}.json")
    image_dir = os.path.join(output_dir, f"{output_name}_images")
    
    # Check if dataset already exists
    if os.path.exists(annotation_path) and os.path.exists(image_dir):
        return annotation_path, image_dir
    
    # Import here to avoid circular imports and unnecessary dependencies
    try:
        from .download_and_convert import download_and_convert_ocr_dataset
        print(f"\n{'='*80}")
        print("🚀 Auto-downloading and converting OCR dataset...")
        print(f"{'='*80}\n")
        
        annotation_path, image_dir = download_and_convert_ocr_dataset(
            hf_dataset_id="kavinh07/nid-synth-200k-ocr",
            output_dir=output_dir,
            output_name=output_name,
            ocr_prompt="OCR the text. target language Bengali & English.",
            max_samples=max_samples,
            split="train"
        )
        return annotation_path, image_dir
    except ImportError as e:
        print(f"⚠️  Could not download OCR dataset: {e}")
        print("   Please run: python -m qwenvl.data.download_and_convert --help")
        return None, None


# Auto-setup OCR dataset (uncomment to enable)
# annotation_path, image_dir = _setup_ocr_dataset_if_needed()
# if annotation_path:
#     OCR_DATASET = {
#         "annotation_path": annotation_path,
#         "data_path": image_dir,
#     }
# else:
OCR_DATASET = {
    "annotation_path": "dataset/ocr_dataset.json",
    "data_path": "dataset/ocr_dataset_images",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "dataset": dataset,
    "ocr_dataset": OCR_DATASET,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["dataset"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
