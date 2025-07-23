#!/usr/bin/env python3
"""
Main training script for Arabic image captioning
"""

import os
import sys
import argparse
from finetune_trainer import ArabicImageCaptionTrainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Fine-tune AraGPT2 for Arabic image captioning")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for data and outputs")
    parser.add_argument("--conservative", action="store_true", help="Use conservative settings for limited VRAM")
    parser.add_argument("--resume_latest", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--config_path", type=str, help="Path to existing config file")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ArabicImageCaptionTrainer(base_dir=args.base_dir)
    
    # Check if we need to create config or use existing one
    if args.config_path and os.path.exists(args.config_path):
        config_path = args.config_path
        print(f"Using existing config: {config_path}")
    else:
        # Check for existing config files
        config_suffix = "conservative" if args.conservative else "standard"
        config_path = os.path.join(args.base_dir, f"aragpt2_arabic_{config_suffix}.yaml")
        
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            print("Run setup_training.py first!")
            
            # Try to auto-setup
            print("Attempting auto-setup...")
            
            # Setup environment
            if not trainer.setup_environment():
                print("❌ Environment setup failed")
                sys.exit(1)
            
            # Try to prepare dataset with default paths
            excel_file = os.path.join(args.base_dir, "Train", "TrainSubtask2.xlsx")
            images_dir = os.path.join(args.base_dir, "Train", "images")
            
            # For Kaggle, try alternative paths
            if not os.path.exists(excel_file):
                excel_file = "/kaggle/input/arabic-dataset/TrainSubtask2.xlsx"  # Update with your dataset name
            if not os.path.exists(images_dir):
                images_dir = "/kaggle/input/arabic-dataset/images"  # Update with your dataset name
            
            if os.path.exists(excel_file) and os.path.exists(images_dir):
                if trainer.prepare_dataset(excel_file=excel_file, images_dir=images_dir):
                    config_path = trainer.create_training_config(conservative=args.conservative)
                    print(f"✅ Auto-setup complete! Created config: {config_path}")
                else:
                    print("❌ Dataset preparation failed")
                    sys.exit(1)
            else:
                print(f"❌ Required files not found:")
                print(f"  Excel: {excel_file}")
                print(f"  Images: {images_dir}")
                print("Please upload your dataset first!")
                sys.exit(1)
    
    # Add resume capability if requested
    if args.resume_latest:
        # Check if config already has resume setting
        with open(config_path, 'r') as f:
            content = f.read()
        
        if 'resume_from_checkpoint:' not in content:
            # Add resume setting
            content = content.replace('### train', '### train\nresume_from_checkpoint: true')
            with open(config_path, 'w') as f:
                f.write(content)
            print("✅ Resume from checkpoint enabled")
    
    # Start training
    print(f"Starting training with config: {config_path}")
    success = trainer.start_training(config_path)
    
    if success:
        print("✅ Training completed successfully!")
    else:
        print("❌ Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()