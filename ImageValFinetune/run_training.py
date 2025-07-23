#!/usr/bin/env python3
"""
Main training script for Custom Arabic Flamingo
"""

import os
import sys
import argparse
import finetune_config as config

def main():
    """Main training function for Custom Arabic Flamingo."""
    parser = argparse.ArgumentParser(description="Train Custom Arabic Flamingo with AraGPT2-Mega")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for data and outputs")
    parser.add_argument("--conservative", action="store_true", help="Use conservative settings for limited VRAM")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    
    args = parser.parse_args()
    
    print("🔥 Custom Arabic Flamingo Training with AraGPT2-Mega 🔥")
    
    # Check if custom Flamingo files exist
    required_files = ['arabic_flamingo_model.py', 'flamingo_trainer.py', 'prepare_flamingo_dataset.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please ensure all custom Flamingo files are present")
        sys.exit(1)
    
    # Import custom trainer
    try:
        from flamingo_trainer import ArabicFlamingoTrainer
        print("✅ Custom Flamingo trainer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import custom Flamingo trainer: {e}")
        print("Make sure arabic_flamingo_model.py and flamingo_trainer.py are properly implemented")
        sys.exit(1)
    
    # Check if dataset needs to be prepared
    dataset_path = os.path.join(args.base_dir, config.DATASET_CONFIG["name"] + ".json")
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("Attempting to prepare dataset...")
        
        # Try to prepare dataset automatically
        try:
            from prepare_flamingo_dataset import prepare_flamingo_dataset
            
            # Update paths for current environment
            excel_file = config.DEFAULT_PATHS["train_excel"].replace("/kaggle/input/your-dataset", args.base_dir)
            images_dir = config.DEFAULT_PATHS["train_images_dir"].replace("/kaggle/input/your-dataset", args.base_dir)
            
            # Check if source files exist
            if os.path.exists(excel_file) and os.path.exists(images_dir):
                prepare_flamingo_dataset(excel_file, images_dir, dataset_path)
                print("✅ Dataset prepared successfully")
            else:
                print(f"❌ Source files not found:")
                print(f"  Excel: {excel_file}")
                print(f"  Images: {images_dir}")
                print("Please upload your dataset first!")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to prepare dataset: {e}")
            sys.exit(1)
    
    # Initialize trainer
    try:
        trainer = ArabicFlamingoTrainer(
            base_dir=args.base_dir,
            model_name=config.DEFAULT_MODEL_NAME
        )
        print("✅ Trainer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize trainer: {e}")
        sys.exit(1)
    
    # Choose config
    train_config = config.CONSERVATIVE_CONFIG if args.conservative else config.TRAINING_CONFIG
    
    print(f"📊 Training configuration:")
    print(f"  - Model: {config.DEFAULT_MODEL_NAME}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {train_config['learning_rate']}")
    print(f"  - Conservative mode: {args.conservative}")
    
    # Start training
    try:
        trainer.train(
            dataset_path=dataset_path,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=train_config["learning_rate"]
        )
        print("🎉 Custom Arabic Flamingo training completed!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()