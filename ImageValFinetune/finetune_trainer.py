"""
Fine-tuning trainer class for Arabic image captioning
"""

import os
import subprocess
import json
import pandas as pd
from typing import Optional, List, Dict
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from tqdm import tqdm

import finetune_config as config
import finetune_utils as utils


class ArabicImageCaptionTrainer:
    """Class for fine-tuning Qwen2.5-VL model for Arabic image captioning."""
    
    def __init__(
        self,
        base_dir: str = config.DEFAULT_PATHS["base_dir"],
        llamafactory_path: str = config.DEFAULT_PATHS["llamafactory_repo"]
    ):
        """
        Initialize the trainer.
        
        Args:
            base_dir: Base directory for data and outputs
            llamafactory_path: Path to LlamaFactory repository
        """
        self.base_dir = base_dir
        self.llamafactory_path = llamafactory_path
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
        images_dir: Optional[str] = None,
        dataset_name: str = config.DATASET_CONFIG["name"]
    ) -> bool:
        """
        Prepare the training dataset.
        
        Args:
            excel_file: Path to Excel file with training data
            images_dir: Directory containing training images
            dataset_name: Name for the dataset
        """
        print("=== Preparing Training Dataset ===")
        
        # Use default paths if not provided
        excel_file = excel_file or self.paths["excel_file"]
        images_dir = images_dir or self.paths["images_dir"]
        
        # Validate inputs
        if not utils.validate_excel_file(excel_file):
            return False
        
        if not os.path.exists(images_dir):
            print(f"❌ Images directory not found: {images_dir}")
            return False
        
        # Create training dataset JSON
        json_path = os.path.join(self.base_dir, "llamafactory_training_data.json")
        if not utils.create_training_dataset(excel_file, images_dir, json_path):
            return False
        
        # Register dataset in LlamaFactory
        if not utils.register_dataset_in_llamafactory(
            dataset_name, json_path, self.llamafactory_path
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
        """
        Create training configuration file.
        
        Args:
            output_dir: Directory to save model checkpoints
            conservative: Use conservative settings for limited VRAM
            custom_config: Custom configuration overrides
            
        Returns:
            Path to created config file
        """
        print("=== Creating Training Configuration ===")
        
        output_dir = output_dir or self.paths["output_dir"]
        
        # Create config filename
        config_suffix = "conservative" if conservative else "standard"
        config_path = os.path.join(self.base_dir, f"qwen_arabic_{config_suffix}.yaml")
        
        # Create configuration
        utils.create_training_config(
            config_path,
            output_dir=output_dir,
            conservative=conservative
        )
        
        # Apply custom config if provided
        if custom_config:
            self._update_config_file(config_path, custom_config)
        
        utils.print_training_summary(config_path)
        return config_path
    
    def _update_config_file(self, config_path: str, updates: Dict):
        """Update configuration file with custom settings."""
        with open(config_path, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            for key, value in updates.items():
                if line.strip().startswith(f"{key}:"):
                    lines[i] = f"{key}: {value}"
        
        with open(config_path, 'w') as f:
            f.write('\n'.join(lines))
    
    def start_training(self, config_path: str) -> bool:
        """
        Start the training process.
        
        Args:
            config_path: Path to training configuration file
        """
        print("=== Starting Training ===")
        
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False
        
        try:
            # Change to LlamaFactory directory
            original_cwd = os.getcwd()
            os.chdir(self.llamafactory_path)
            
            # Start training
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
    
    def list_checkpoints(self) -> List[str]:
        """List available model checkpoints."""
        checkpoints = utils.get_available_checkpoints(self.paths["output_dir"])
        
        if checkpoints:
            print(f"Available checkpoints in {self.paths['output_dir']}:")
            for cp in checkpoints:
                print(f"  - {cp}")
        else:
            print("No checkpoints found")
        
        return checkpoints
    
    def evaluate_model(
        self,
        checkpoint_path: Optional[str] = None,
        test_images_dir: Optional[str] = None,
        max_images: Optional[int] = None
    ) -> List[Dict]:
        """
        Evaluate the fine-tuned model on test images.
        
        Args:
            checkpoint_path: Path to model checkpoint
            test_images_dir: Directory containing test images
            max_images: Maximum number of images to process
            
        Returns:
            List of evaluation results
        """
        print("=== Evaluating Fine-tuned Model ===")
        
        # Use default paths if not provided
        test_images_dir = test_images_dir or self.paths["test_images_dir"]
        
        if not checkpoint_path:
            checkpoints = self.list_checkpoints()
            if not checkpoints:
                print("❌ No checkpoints available for evaluation")
                return []
            checkpoint_path = os.path.join(self.paths["output_dir"], checkpoints[-1])
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return []
        
        if not os.path.exists(test_images_dir):
            print(f"❌ Test images directory not found: {test_images_dir}")
            return []
        
        # Load model and processor
        try:
            print("Loading fine-tuned model...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            processor = AutoProcessor.from_pretrained(config.DEFAULT_MODEL_NAME)
            print("✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return []
        
        # Get test images
        image_files = []
        for file in os.listdir(test_images_dir):
            if any(file.lower().endswith(ext) for ext in config.SUPPORTED_IMAGE_FORMATS):
                image_files.append(file)
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"Processing {len(image_files)} test images...")
        
        # Process images
        results = []
        for i, image_file in enumerate(tqdm(image_files, desc="Generating captions")):
            try:
                result = self._process_single_image(
                    os.path.join(test_images_dir, image_file),
                    image_file,
                    model,
                    processor
                )
                results.append(result)
                
                # Print progress for first few images
                if i < 5:
                    print(f"\n{image_file}: {result['arabic_caption']}")
                    
            except Exception as e:
                print(f"❌ Error processing {image_file}: {e}")
                results.append({
                    "image_file": image_file,
                    "arabic_caption": f"Error: {str(e)}"
                })
        
        # Save results
        self._save_evaluation_results(results)
        
        print(f"\n✅ Evaluation complete: {len(results)} images processed")
        successful = len([r for r in results if not r['arabic_caption'].startswith('Error:')])
        print(f"Successful: {successful}, Failed: {len(results) - successful}")
        
        return results
    
    def _process_single_image(
        self,
        image_path: str,
        image_file: str,
        model,
        processor
    ) -> Dict:
        """Process a single image and generate caption."""
        image = Image.open(image_path)
        
        # Create prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image in Arabic."}
                ]
            }
        ]
        
        # Process and generate
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                pad_token_id=processor.tokenizer.eos_token_id,
                **config.GENERATION_CONFIG
            )
        
        response = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract caption
        if "assistant\n" in response:
            arabic_caption = response.split("assistant\n")[-1].strip()
        else:
            arabic_caption = response.split("Describe this image in Arabic.")[-1].strip()
        
        return {
            "image_file": image_file,
            "image_path": image_path,
            "arabic_caption": arabic_caption
        }
    
    def _save_evaluation_results(self, results: List[Dict]):
        """Save evaluation results to JSON and CSV."""
        # Save to JSON
        json_output = os.path.join(self.base_dir, "generated_arabic_captions.json")
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Save to CSV
        csv_output = os.path.join(self.base_dir, "fine_tune_generated_arabic_captions.csv")
        df = pd.DataFrame(results)
        df.to_csv(csv_output, index=False, encoding='utf-8-sig')
        
        print(f"Results saved to:")
        print(f"  JSON: {json_output}")
        print(f"  CSV: {csv_output}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL for Arabic image captioning")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for data and outputs")
    parser.add_argument("--excel_file", type=str, help="Path to Excel training file")
    parser.add_argument("--images_dir", type=str, help="Directory containing training images")
    parser.add_argument("--conservative", action="store_true", help="Use conservative settings for limited VRAM")
    parser.add_argument("--skip_setup", action="store_true", help="Skip environment setup")
    parser.add_argument("--skip_training", action="store_true", help="Skip training, only prepare dataset")
    parser.add_argument("--evaluate_only", action="store_true", help="Only run evaluation on existing checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint path for evaluation")
    parser.add_argument("--max_eval_images", type=int, help="Maximum number of images to evaluate")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ArabicImageCaptionTrainer(base_dir=args.base_dir)
    
    if args.evaluate_only:
        # Only run evaluation
        trainer.evaluate_model(
            checkpoint_path=args.checkpoint,
            max_images=args.max_eval_images
        )
        return
    
    # Setup environment
    if not args.skip_setup:
        if not trainer.setup_environment():
            print("❌ Environment setup failed")
            return
    
    # Prepare dataset
    if not trainer.prepare_dataset(
        excel_file=args.excel_file,
        images_dir=args.images_dir
    ):
        print("❌ Dataset preparation failed")
        return
    
    if args.skip_training:
        print("✅ Dataset preparation complete. Skipping training.")
        return
    
    # Create training configuration
    config_path = trainer.create_training_config(conservative=args.conservative)
    
    # Start training
    if trainer.start_training(config_path):
        print("\n✅ Training completed successfully!")
        
        # Run evaluation on the trained model
        print("\nRunning evaluation on trained model...")
        trainer.evaluate_model(max_images=args.max_eval_images)
    else:
        print("❌ Training failed")


if __name__ == "__main__":
    main()