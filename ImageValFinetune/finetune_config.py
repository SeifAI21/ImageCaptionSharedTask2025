"""
Configuration file for Custom Arabic Flamingo with AraGPT2-Mega
"""

import os

# Model configuration - Custom Flamingo with AraGPT2
DEFAULT_MODEL_NAME = "aubmindlab/aragpt2-mega"  # Language model
VISION_MODEL_NAME = "openai/clip-vit-large-patch14"  # Vision encoder
IMAGE_SIZE = 224

# Flamingo-specific configuration
FLAMINGO_CONFIG = {
    "vision_encoder_path": VISION_MODEL_NAME,
    "lang_model_path": DEFAULT_MODEL_NAME,
    "cross_attn_every_n_layers": 4,
    "perceiver_num_latents": 64,
    "perceiver_depth": 6,
    "vision_dim": 1024,  # CLIP-Large dimension
    "lang_dim": 1536,    # AraGPT2-mega dimension
}

# Training configuration for Custom Flamingo
TRAINING_CONFIG = {
    # Model settings
    "model_type": "custom_flamingo",
    "use_custom_trainer": True,  # Use our custom trainer, not LlamaFactory
    
    # LoRA settings
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "lora_target": "c_attn,c_proj,c_fc",  # ADDED: Missing lora_target
    
    # Training parameters
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 5.0e-6,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    
    # Training settings
    "fp16": True,
    "gradient_clipping": 1.0,
    "save_steps": 50,
    "eval_steps": 100,
    "logging_steps": 10,
}

# Conservative settings - FIXED for anti-overfitting
CONSERVATIVE_CONFIG = TRAINING_CONFIG.copy()  # Start with base config
CONSERVATIVE_CONFIG.update({
    "lora_rank": 4,
    "lora_alpha": 8,
    "lora_dropout": 0.3,
    "lora_target": "c_attn,c_proj",        # ADDED: Fewer targets for conservative
    "gradient_accumulation_steps": 64,     # INCREASED from 32
    "learning_rate": 5e-7,                 # Much lower to prevent overfitting
    "batch_size": 1,                       # KEEP at 1
    "weight_decay": 0.1,                   # Higher regularization
    "warmup_ratio": 0.3,
    
    # Memory optimization
    "fp16": True,
    "gradient_checkpointing": True,
    
    # Anti-overfitting settings
    "save_steps": 500,                     # Less frequent saves
    "eval_steps": 100,
    "logging_steps": 10,
    "gradient_clipping": 0.5,              # Stronger clipping
})

# Dataset configuration for Custom Flamingo - FIXED
DATASET_CONFIG = {
    "name": "arabic_flamingo_dataset",
    "template": "default",              # ADDED: Missing template key
    "image_token": "<image>",
    "prompt_template": "<image> وصف هذه الصورة:",
    "max_length": 512,
    "cutoff_len": 512,                  # ADDED: For compatibility
    "overwrite_cache": True,            # ADDED: For compatibility
    "preprocessing_num_workers": 1,     # ADDED: For compatibility
    "dataloader_num_workers": 0,        # ADDED: For compatibility
}

# Default paths - FIXED for Kaggle
DEFAULT_PATHS = {
    "base_dir": "/kaggle/working",
    "llamafactory_repo": "/kaggle/working/LLaMA-Factory",
    "train_excel": "/kaggle/input/trainsubtask2/TrainSubtask2.xlsx",
    "train_images_dir": "/kaggle/working/Train/images",
    "test_images_dir": "/kaggle/working/Test/images",
    "output_dir": "/kaggle/working/arabic_flamingo_model",
    "dataset_json": "/kaggle/working/arabic_flamingo_dataset.json"
}

SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# IMAGE_MAX_PIXELS for compatibility
IMAGE_MAX_PIXELS = 224 * 224

# YAML template - FIXED with proper placeholders
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

# Additional YAML template variables that might be needed
YAML_DEFAULTS = {
    "plot_loss": True,
    "overwrite_output_dir": True,
    "save_only_model": True,
    "report_to": "none",
    "lr_scheduler_type": "cosine",
    "val_size": 0.1,
    "per_device_eval_batch_size": 1,
    "eval_strategy": "steps",
}