"""
Configuration file for Arabic Flamingo with AraGPT2
"""

import os

# Model configuration - Custom Flamingo with AraGPT2
DEFAULT_MODEL_NAME = "aubmindlab/aragpt2-mega"  # or "aubmindlab/aragpt2-large"
VISION_MODEL_NAME = "openai/clip-vit-large-patch14"
IMAGE_MAX_PIXELS = 224 * 224

# Flamingo-specific configuration
FLAMINGO_CONFIG = {
    "vision_encoder": VISION_MODEL_NAME,
    "language_model": DEFAULT_MODEL_NAME,
    "cross_attn_every_n_layers": 4,
    "perceiver_num_latents": 64,
    "perceiver_depth": 6,
    "vision_dim": 1024,  # CLIP-Large dimension
    "lang_dim": 1024,    # AraGPT2-mega dimension
}

# Training configuration for Flamingo+AraGPT2
TRAINING_CONFIG = {
    # LoRA settings
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "lora_target": "all",
    
    # Training parameters (adjusted for Arabic GPT)
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 5.0e-6,  # Lower for pre-trained Arabic model
    "num_train_epochs": 10.0,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "fp16": True,
    "gradient_checkpointing": True,
    
    # Evaluation
    "val_size": 0.2,
    "per_device_eval_batch_size": 1,
    "eval_strategy": "steps",
    "eval_steps": 15,
    
    # Logging and saving
    "logging_steps": 5,
    "save_steps": 30,
    "plot_loss": True,
    "overwrite_output_dir": True,
    "save_only_model": False,
    "report_to": "none",
    
    # Data processing (adjusted for Arabic)
    "cutoff_len": 512,  # Shorter for AraGPT2
    "overwrite_cache": True,
    "preprocessing_num_workers": 2,
    "dataloader_num_workers": 0,
}

# Conservative settings
CONSERVATIVE_CONFIG = TRAINING_CONFIG.copy()
CONSERVATIVE_CONFIG.update({
    "lora_rank": 4,
    "gradient_accumulation_steps": 32,
    "learning_rate": 2.0e-6,
    "per_device_train_batch_size": 1,
})

# Dataset configuration for Arabic Flamingo
DATASET_CONFIG = {
    "name": "arabic_captions_flamingo_aragpt2",
    "template": "default",  # Custom template for AraGPT2
    "conversation_template": {
        "human_prefix": "",
        "assistant_prefix": "",
        "system_message": "",
        "user_prompt": "<صورة> وصف هذه الصورة:",  # Arabic image token
        "response_format": "{caption}"
    }
}

# Default paths
DEFAULT_PATHS = {
    "base_dir": "/content/drive/MyDrive/ImageVal",
    "llamafactory_repo": "/content/LLaMA-Factory",
    "train_excel": "/content/drive/MyDrive/ImageVal/Train/TrainSubtask2.xlsx",
    "train_images_dir": "/content/drive/MyDrive/ImageVal/Train/images",
    "test_images_dir": "/content/drive/MyDrive/ImageVal/Test/images",
    "output_dir": "/content/drive/MyDrive/ImageVal/flamingo_aragpt2_model",
    "dataset_json": "/content/drive/MyDrive/ImageVal/arabic_captions_flamingo_aragpt2.json"
}

SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# YAML template for LlamaFactory - UPDATED FOR FLAMINGO
YAML_TEMPLATE = """### model
model_name_or_path: {model_name}
template: {template}
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

### wandb
run_name: {run_name}
"""