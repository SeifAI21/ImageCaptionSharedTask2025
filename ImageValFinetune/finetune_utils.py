"""
Utility functions for fine-tuning setup and data processing
"""

import os
import json
import pandas as pd
import subprocess
import torch
from PIL import Image
from typing import List, Dict, Optional, Tuple
import finetune_config as config


def check_system_requirements():
    """Check GPU availability and system requirements."""
    print("=== System Requirements Check ===")

    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")

        total_memory = 0
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            total_memory += gpu_memory

        print(f"Total GPU Memory: {total_memory:.1f} GB")

        if num_gpus > 1:
            print("✅ Multi-GPU setup detected - will use distributed training")
        elif total_memory < 12:
            print("⚠️ Warning: Less than 12GB VRAM. Consider using conservative settings.")
        else:
            print("✅ Sufficient GPU memory for training")
    else:
        print("❌ CUDA not available. GPU required for fine-tuning.")
        return False

    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    return True


def install_llamafactory(repo_path: str = "/content/LLaMA-Factory"):
    """Install LlamaFactory from GitHub."""
    print("=== Installing LlamaFactory ===")
    
    if os.path.exists(repo_path):
        print(f"LlamaFactory already exists at {repo_path}")
        return True
    
    try:
        # Clone repository
        subprocess.run([
            "git", "clone", "--depth", "1", 
            "https://github.com/hiyouga/LLaMA-Factory.git", 
            repo_path
        ], check=True)
        
        # Change to repo directory and install
        original_cwd = os.getcwd()
        os.chdir(repo_path)
        
        subprocess.run([
            "pip", "install", "-e", ".[torch,metrics]"
        ], check=True)
        
        os.chdir(original_cwd)
        
        print("✅ LlamaFactory installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install LlamaFactory: {e}")
        return False
    
def get_timestamp() -> str:
    """Get current timestamp string."""
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def create_directory_structure(base_dir: str):
    """Create necessary directories for training."""
    directories = [
        base_dir,
        os.path.join(base_dir, "Train"),
        os.path.join(base_dir, "Train", "images"),
        os.path.join(base_dir, "Test"),
        os.path.join(base_dir, "Test", "images"),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def validate_excel_file(excel_path: str) -> bool:
    """Validate the Excel training file."""
    if not os.path.exists(excel_path):
        print(f"❌ Excel file not found: {excel_path}")
        return False
    
    try:
        df = pd.read_excel(excel_path)
        required_columns = ['File Name', 'Description']
        
        for col in required_columns:
            if col not in df.columns:
                print(f"❌ Missing required column: {col}")
                return False
        
        # Check for non-empty rows
        valid_rows = df.dropna(subset=['File Name', 'Description'])
        print(f"✅ Excel file valid: {len(valid_rows)} training examples found")
        return True
        
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        return False


def validate_images(image_dir: str, required_files: List[str]) -> Tuple[List[str], List[str]]:
    """Validate image files exist and are readable."""
    found_images = []
    missing_images = []
    
    for filename in required_files:
        # Try different extensions
        for ext in ['.jpg', '.jpeg', '.png']:
            full_path = os.path.join(image_dir, f"{filename}{ext}")
            if os.path.exists(full_path):
                try:
                    with Image.open(full_path) as img:
                        img.verify()
                    found_images.append(f"{filename}{ext}")
                    break
                except Exception as e:
                    print(f"⚠️ Corrupted image: {full_path} - {e}")
        else:
            missing_images.append(filename)
    
    print(f"Found {len(found_images)} valid images")
    print(f"Missing {len(missing_images)} images")
    
    if missing_images and len(missing_images) <= 5:
        print("Missing files:")
        for img in missing_images:
            print(f"  - {img}")
    
    return found_images, missing_images


def create_training_dataset(
    excel_path: str, 
    images_dir: str, 
    output_path: str,
    use_absolute_paths: bool = True
) -> bool:
    """Create training dataset in LlamaFactory format."""
    print("=== Creating Training Dataset ===")
    
    try:
        # Read Excel file
        df = pd.read_excel(excel_path)
        
        # Get valid image files
        image_filenames = df['File Name'].dropna().tolist()
        found_images, missing_images = validate_images(images_dir, image_filenames)
        
        # Create training data
        training_data = []
        
        for _, row in df.iterrows():
            if pd.notna(row['File Name']) and pd.notna(row['Description']):
                filename = row['File Name']
                
                # Check if we have this image
                matching_images = [img for img in found_images if img.startswith(filename)]
                if not matching_images:
                    continue
                
                image_file = matching_images[0]
                
                if use_absolute_paths:
                    image_path = os.path.join(images_dir, image_file)
                else:
                    image_path = f"images/{image_file}"
                
                entry = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>Describe this image in Arabic."
                        },
                        {
                            "from": "gpt",
                            "value": str(row['Description'])
                        }
                    ],
                    "images": [image_path]
                }
                training_data.append(entry)
        
        # Save training data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Created {len(training_data)} training examples")
        print(f"Saved to: {output_path}")
        
        return len(training_data) > 0
        
    except Exception as e:
        print(f"❌ Error creating training dataset: {e}")
        return False


def register_dataset_in_llamafactory(
    dataset_name: str,
    json_path: str,
    llamafactory_path: str
) -> bool:
    """Register dataset in LlamaFactory's dataset configuration."""
    try:
        # Path to LlamaFactory's dataset info file
        dataset_info_path = os.path.join(llamafactory_path, "data", "dataset_info.json")
        
        # Load existing dataset info
        if os.path.exists(dataset_info_path):
            with open(dataset_info_path, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)
        else:
            dataset_info = {}
        
        # Add our dataset
        dataset_info[dataset_name] = {
            "file_name": json_path,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt"
            }
        }
        
        # Save updated dataset info
        os.makedirs(os.path.dirname(dataset_info_path), exist_ok=True)
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Dataset '{dataset_name}' registered in LlamaFactory")
        return True
        
    except Exception as e:
        print(f"❌ Error registering dataset: {e}")
        return False


# Replace the entire create_training_config method with this:

def create_training_config(self, output_dir=None, conservative=True, custom_config=None):
    """Create training configuration YAML with ALL required variables"""
    
    if output_dir is None:
        output_dir = os.path.join(self.base_dir, "output")
    
    # Select configuration
    if custom_config:
        train_config = custom_config
    elif conservative:
        train_config = config.CONSERVATIVE_CONFIG.copy()
    else:
        train_config = config.TRAINING_CONFIG.copy()
    
    # Ensure all required keys exist by merging with base config
    base_config = config.TRAINING_CONFIG.copy()
    base_config.update(train_config)
    train_config = base_config
    
    # Prepare ALL required variables for YAML template
    yaml_vars = {
        # Model settings
        "model_name": config.DEFAULT_MODEL_NAME,
        "template": config.DATASET_CONFIG.get("template", "default"),
        
        # LoRA settings
        "lora_rank": train_config.get("lora_rank", 4),
        "lora_alpha": train_config.get("lora_alpha", 8),
        "lora_dropout": train_config.get("lora_dropout", 0.3),
        "lora_target": train_config.get("lora_target", "c_attn"),
        
        # Dataset settings
        "dataset_name": config.DATASET_CONFIG.get("name", "arabic_flamingo_dataset"),
        "cutoff_len": config.DATASET_CONFIG.get("cutoff_len", 512),
        "overwrite_cache": config.DATASET_CONFIG.get("overwrite_cache", True),
        "preprocessing_num_workers": config.DATASET_CONFIG.get("preprocessing_num_workers", 1),
        "dataloader_num_workers": config.DATASET_CONFIG.get("dataloader_num_workers", 0),
        
        # Output settings
        "output_dir": output_dir,
        "logging_steps": train_config.get("logging_steps", 10),
        "save_steps": train_config.get("save_steps", 500),
        "plot_loss": config.YAML_DEFAULTS.get("plot_loss", True),
        "overwrite_output_dir": config.YAML_DEFAULTS.get("overwrite_output_dir", True),
        "save_only_model": config.YAML_DEFAULTS.get("save_only_model", True),
        "report_to": config.YAML_DEFAULTS.get("report_to", "none"),
        
        # Training settings
        "per_device_train_batch_size": train_config.get("batch_size", 1),
        "gradient_accumulation_steps": train_config.get("gradient_accumulation_steps", 32),
        "learning_rate": train_config.get("learning_rate", 1e-7),
        "num_train_epochs": train_config.get("num_epochs", 1),
        "lr_scheduler_type": config.YAML_DEFAULTS.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": train_config.get("warmup_ratio", 0.1),
        "fp16": train_config.get("fp16", True),
        "gradient_checkpointing": train_config.get("gradient_checkpointing", True),
        
        # Evaluation settings
        "val_size": config.YAML_DEFAULTS.get("val_size", 0.1),
        "per_device_eval_batch_size": config.YAML_DEFAULTS.get("per_device_eval_batch_size", 1),
        "eval_strategy": config.YAML_DEFAULTS.get("eval_strategy", "steps"),
        "eval_steps": train_config.get("eval_steps", 100),
        
        # Wandb settings
        "run_name": f"arabic_flamingo_{'conservative' if conservative else 'standard'}"
    }
    
    try:
        # Format the YAML template with all variables
        yaml_content = config.YAML_TEMPLATE.format(**yaml_vars)
        return yaml_content
    except KeyError as e:
        print(f"❌ Missing YAML variable: {e}")
        print("Available variables:", list(yaml_vars.keys()))
        raise


def get_available_checkpoints(model_dir: str) -> List[str]:
    """Get list of available model checkpoints."""
    if not os.path.exists(model_dir):
        return []
    
    checkpoints = []
    for item in os.listdir(model_dir):
        if item.startswith('checkpoint-') and os.path.isdir(os.path.join(model_dir, item)):
            checkpoints.append(item)
    
    return sorted(checkpoints, key=lambda x: int(x.split('-')[1]))


def print_training_summary(config_path: str):
    """Print training configuration summary."""
    print("\n=== Training Configuration Summary ===")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            content = f.read()
            
        print(f"Config file: {config_path}")
        print("\nKey settings:")
        
        # Extract key information
        lines = content.split('\n')
        key_settings = [
            'model_name_or_path', 'lora_rank', 'learning_rate', 
            'num_train_epochs', 'per_device_train_batch_size',
            'gradient_accumulation_steps', 'output_dir'
        ]
        
        for line in lines:
            for setting in key_settings:
                if line.strip().startswith(f"{setting}:"):
                    print(f"  {line.strip()}")
    else:
        print(f"Config file not found: {config_path}")

def create_training_dataset_flamingo(
    excel_path: str, 
    images_dir: str, 
    output_path: str,
    use_absolute_paths: bool = True
) -> bool:
    """Create training dataset in Flamingo format."""
    print("=== Creating Flamingo Training Dataset ===")
    
    try:
        df = pd.read_excel(excel_path)
        found_images, missing_images = validate_images(images_dir, df['File Name'].dropna().tolist())
        
        training_data = []
        
        for _, row in df.iterrows():
            if pd.notna(row['File Name']) and pd.notna(row['Description']):
                filename = row['File Name']
                matching_images = [img for img in found_images if img.startswith(filename)]
                
                if not matching_images:
                    continue
                
                image_file = matching_images[0]
                image_path = os.path.join(images_dir, image_file) if use_absolute_paths else f"images/{image_file}"
                
                # Flamingo format
                entry = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>وصف هذه الصورة باللغة العربية."
                        },
                        {
                            "from": "gpt", 
                            "value": str(row['Description'])
                        }
                    ],
                    "images": [image_path]
                }
                training_data.append(entry)
        
        # Save training data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Created {len(training_data)} Flamingo training examples")
        return len(training_data) > 0
        
    except Exception as e:
        print(f"❌ Error creating Flamingo dataset: {e}")
        return False