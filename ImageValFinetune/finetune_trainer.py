"""
Arabic Image Caption Trainer - Working Implementation
"""
import os
import json
import subprocess
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from typing import Optional, List, Dict
import finetune_config as config
import finetune_utils as utils

class ArabicImageCaptionTrainer:
    """Class for fine-tuning models for Arabic image captioning."""
    
    def __init__(
        self,
        base_dir: str = config.DEFAULT_PATHS["base_dir"],
        llamafactory_path: str = config.DEFAULT_PATHS["llamafactory_repo"],
        wandb_project: str = "arabic-image-captioning",
        wandb_entity: Optional[str] = None
    ):
        self.base_dir = base_dir
        self.llamafactory_path = llamafactory_path
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.paths = config.DEFAULT_PATHS.copy()
        self.paths["base_dir"] = base_dir
        
        # Update other paths based on base_dir
        for key, path in self.paths.items():
            if key != "llamafactory_repo" and "drive/MyDrive/ImageVal" in path:
                self.paths[key] = path.replace("/content/drive/MyDrive/ImageVal", base_dir)
    
    def setup_environment(self) -> bool:
        """Set up the training environment."""
        print("=== Setting Up Training Environment ===")
        
        # Check system requirements
        if not utils.check_system_requirements():
            return False
        
        # Install LlamaFactory
        if not utils.install_llamafactory(self.llamafactory_path):
            return False
        
        # Create directory structure
        utils.create_directory_structure(self.base_dir)
        
        print("✅ Environment setup complete")
        return True
    
    def prepare_dataset(
        self,
        excel_file: Optional[str] = None,
        images_dir: Optional[str] = None
    ) -> bool:
        """Prepare dataset for training."""
        print("=== Preparing Dataset ===")
        
        excel_file = excel_file or self.paths["train_excel"]
        images_dir = images_dir or self.paths["train_images_dir"]
        
        # Validate inputs
        if not utils.validate_excel_file(excel_file):
            return False
        
        # Create training dataset
        dataset_path = self.paths["dataset_json"]
        if not utils.create_training_dataset_flamingo(excel_file, images_dir, dataset_path):
            return False
        
        # Register dataset in LlamaFactory
        if not utils.register_dataset_in_llamafactory(
            config.DATASET_CONFIG["name"],
            dataset_path,
            self.llamafactory_path
        ):
            return False
        
        print("✅ Dataset preparation complete")
        return True
    
    def create_training_config(
        self,
        output_dir: Optional[str] = None,
        conservative: bool = False,
        custom_config: Optional[Dict] = None
    ) -> str:
        """Create training configuration file."""
        print("=== Creating Training Configuration ===")
        
        output_dir = output_dir or self.paths["output_dir"]
        
        # Create config filename
        config_suffix = "conservative" if conservative else "standard"
        config_path = os.path.join(self.base_dir, f"aragpt2_arabic_{config_suffix}.yaml")
        
        # Choose config based on conservative flag
        train_config = config.CONSERVATIVE_CONFIG if conservative else config.TRAINING_CONFIG
        
        # Generate run name
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config_type = "conservative" if conservative else "standard"
        run_name = f"aragpt2_arabic_{config_type}_{timestamp}"
        
        # Format the YAML template
        yaml_content = config.YAML_TEMPLATE.format(
            model_name=config.DEFAULT_MODEL_NAME,
            dataset_name=config.DATASET_CONFIG["name"],
            template=config.DATASET_CONFIG["template"],
            output_dir=output_dir,
            run_name=run_name,
            **train_config
        )
        
        # Save configuration
        with open(config_path, 'w') as f:
            f.write(yaml_content)
        
        # Apply custom config if provided
        if custom_config:
            self._update_config_file(config_path, custom_config)
        
        utils.print_training_summary(config_path)
        return config_path
    
    def start_training(self, config_path: str, num_gpus: int = None) -> bool:
        """Start the training process."""
        print("=== Starting Training ===")
        
        # Auto-detect number of GPUs if not specified
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        
        print(f"Using {num_gpus} GPU(s) for training")
        
        # Set environment variables for optimization
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # Clear any existing CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False
        
        try:
            # Change to LlamaFactory directory
            original_cwd = os.getcwd()
            os.chdir(self.llamafactory_path)
            
            # Build command based on number of GPUs
            if num_gpus > 1:
                # Use torchrun for multi-GPU training
                cmd = [
                    "torchrun",
                    f"--nproc_per_node={num_gpus}",
                    "--master_port=29501",
                    "-m", "llamafactory.train.tuner",
                    "--config", config_path
                ]
            else:
                # Single GPU training
                cmd = ["llamafactory-cli", "train", config_path]
            
            print(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, check=True, capture_output=False)
            
            os.chdir(original_cwd)
            
            print("✅ Training completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
            os.chdir(original_cwd)
            return False
    
    def evaluate_model(
        self,
        checkpoint_path: Optional[str] = None,
        test_images_dir: Optional[str] = None,
        max_images: Optional[int] = None
    ) -> Optional[List[Dict]]:
        """Evaluate the trained model."""
        print("=== Model Evaluation ===")
        
        # Find checkpoint if not specified
        if not checkpoint_path:
            checkpoint_path = self._find_latest_checkpoint()
            if not checkpoint_path:
                print("❌ No checkpoints found")
                return None
        
        print(f"Using checkpoint: {checkpoint_path}")
        
        # Use default test images directory if not specified
        test_images_dir = test_images_dir or self.paths["test_images_dir"]
        
        if not os.path.exists(test_images_dir):
            print(f"❌ Test images directory not found: {test_images_dir}")
            return None
        
        try:
            # Load model and processor (simplified - for demo)
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            print("Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(config.DEFAULT_MODEL_NAME)
            
            # Get test images
            image_files = [
                f for f in os.listdir(test_images_dir)
                if any(f.lower().endswith(ext) for ext in config.SUPPORTED_IMAGE_FORMATS)
            ]
            
            if max_images:
                image_files = image_files[:max_images]
            
            print(f"Evaluating {len(image_files)} images...")
            
            results = []
            
            for image_file in tqdm(image_files, desc="Generating captions"):
                image_path = os.path.join(test_images_dir, image_file)
                
                try:
                    # Generate caption (simplified)
                    caption = self._generate_caption_simple(model, tokenizer, image_path)
                    
                    result = {
                        'image_file': image_file,
                        'image_path': image_path,
                        'arabic_caption': caption,
                        'timestamp': utils.get_timestamp() if hasattr(utils, 'get_timestamp') else 'unknown'
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                    continue
            
            # Save results
            self._save_evaluation_results(results)
            
            print(f"✅ Evaluation completed! Generated {len(results)} captions")
            return results
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return None
    
    def _generate_caption_simple(self, model, tokenizer, image_path: str) -> str:
        """Generate caption (simplified version for demo)."""
        # This is a simplified version - in practice you'd need proper image processing
        prompt = "وصف هذه الصورة:"
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return caption.replace(prompt, "").strip()
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint."""
        checkpoints = utils.get_available_checkpoints(self.paths["output_dir"])
        if checkpoints:
            return os.path.join(self.paths["output_dir"], checkpoints[-1])
        return None
    
    def _save_evaluation_results(self, results: List[Dict]):
        """Save evaluation results."""
        output_file = os.path.join(self.base_dir, "evaluation_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_file}")
    
    def _update_config_file(self, config_path: str, updates: Dict):
        """Update configuration file with custom settings."""
        with open(config_path, 'r') as f:
            content = f.read()
        
        for key, value in updates.items():
            content = content.replace(f"{key}:", f"{key}: {value}")
        
        with open(config_path, 'w') as f:
            f.write(content)


# Keep your custom Flamingo model for future use
class ArabicFlamingoModel:
    """Custom Flamingo model - for future implementation"""
    
    def __init__(self):
        # Placeholder for future Flamingo implementation
        pass
    
    def generate_caption(self, image_path: str) -> str:
        """Generate caption using custom Flamingo model"""
        # Placeholder - implement later
        return "تم إنشاء الوصف باستخدام نموذج فلامنجو المخصص"