"""
Configuration file for Arabic Image Captioning Fine-tuning
"""

import os

# Model configuration
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_MAX_PIXELS = 131072

# Training configuration
TRAINING_CONFIG = {
    # LoRA settings
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "lora_target": "all",
    
    # Training parameters
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 2.0e-5,
    "num_train_epochs": 15.0,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "fp16": False,
    "gradient_checkpointing": True,
    
    # Evaluation
    "val_size": 0.2,
    "per_device_eval_batch_size": 1,
    "eval_strategy": "steps",
    "eval_steps": 10,
    
    # Logging and saving
    "logging_steps": 5,
    "save_steps": 25,
    "plot_loss": True,
    "overwrite_output_dir": True,
    "save_only_model": False,
    "report_to": "none",
    
    # Data processing
    "cutoff_len": 1024,
    "overwrite_cache": True,
    "preprocessing_num_workers": 2,
    "dataloader_num_workers": 0,
}

# Conservative settings for limited VRAM
CONSERVATIVE_CONFIG = TRAINING_CONFIG.copy()
CONSERVATIVE_CONFIG.update({
    "lora_rank": 4,
    "gradient_accumulation_steps": 32,
    "per_device_train_batch_size": 1,
    "preprocessing_num_workers": 1,
    "dataloader_num_workers": 0,
})

# Paths (adjust these according to your setup)
DEFAULT_PATHS = {
    "base_dir": "/content/drive/MyDrive/ImageVal",
    "train_dir": "/content/drive/MyDrive/ImageVal/Train",
    "test_dir": "/content/drive/MyDrive/ImageVal/Test", 
    "images_dir": "/content/drive/MyDrive/ImageVal/Train/images",
    "test_images_dir": "/content/drive/MyDrive/ImageVal/Test/images",
    "excel_file": "/content/drive/MyDrive/ImageVal/Train/TrainSubtask2.xlsx",
    "output_dir": "/content/drive/MyDrive/ImageVal/qwen2_5vl_arabic_model",
    "llamafactory_repo": "/content/LLaMA-Factory",
}

# Dataset configuration
DATASET_CONFIG = {
    "name": "arabic_captions",
    "template": "qwen2_vl",
    "conversation_template": {
        "human_prefix": "human",
        "gpt_prefix": "gpt",
        "system_message": "You are an expert in visual scene understanding and multilingual caption generation.",
        "user_prompt": "<image>Describe this image in Arabic.",
    }
}

# Supported image formats
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

# Generation settings for inference
GENERATION_CONFIG = {
    "max_new_tokens": 128,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}

# YAML template for LlamaFactory
YAML_TEMPLATE = """### model
model_name_or_path: {model_name}
image_max_pixels: {image_max_pixels}
trust_remote_code: true


### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: {lora_rank}
lora_alpha: {lora_alpha}
lora_dropout: {lora_dropout}
lora_target: {lora_target}

### dataset
dataset: {dataset_name}
template: {template}
cutoff_len: {cutoff_len}
overwrite_cache: {overwrite_cache}
preprocessing_num_workers: {preprocessing_num_workers}
dataloader_num_workers: {dataloader_num_workers}

### output
output_dir: {output_dir}
logging_steps: {logging_steps}
save_steps: {save_steps}
plot_loss: {plot_loss}
overwrite_output_dir: {overwrite_output_dir}
save_only_model: {save_only_model}
report_to: {report_to}

### train
per_device_train_batch_size: {per_device_train_batch_size}
gradient_accumulation_steps: {gradient_accumulation_steps}
learning_rate: {learning_rate}
num_train_epochs: {num_train_epochs}
lr_scheduler_type: {lr_scheduler_type}
warmup_ratio: {warmup_ratio}
fp16: {fp16}
gradient_checkpointing: {gradient_checkpointing}

### eval
val_size: {val_size}
per_device_eval_batch_size: {per_device_eval_batch_size}
eval_strategy: {eval_strategy}
eval_steps: {eval_steps}
"""