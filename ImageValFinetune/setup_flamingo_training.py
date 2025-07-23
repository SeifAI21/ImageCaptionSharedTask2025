#!/usr/bin/env python3
"""
Setup script for Custom Arabic Flamingo training
"""

import os
import sys
import argparse

def setup_custom_flamingo(base_dir: str, excel_file: str, images_dir: str):
    """Setup for Custom Arabic Flamingo training."""
    print("🔥 Custom Arabic Flamingo Setup 🔥")
    
    # Validate inputs
    if not os.path.exists(excel_file):
        print(f"❌ Excel file not found: {excel_file}")
        return False
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return False
    
    # Check required files
    required_files = ['arabic_flamingo_model.py', 'flamingo_trainer.py', 'prepare_flamingo_dataset.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    # Import after checking files exist
    try:
        from prepare_flamingo_dataset import prepare_flamingo_dataset
        import finetune_config as config
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Prepare dataset
    dataset_path = os.path.join(base_dir, "arabic_flamingo_dataset.json")
    
    try:
        prepare_flamingo_dataset(excel_file, images_dir, dataset_path)
        print("✅ Dataset preparation complete")
        
        print(f"\n🎉 Setup complete!")
        print(f"Dataset: {dataset_path}")
        print(f"\nTo start training, run:")
        print(f"python run_training.py --base_dir {base_dir} --conservative")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset preparation failed: {e}")
        return False

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Custom Arabic Flamingo training")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory")
    parser.add_argument("--excel_file", type=str, required=True, help="Path to Excel training file")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing training images")
    
    args = parser.parse_args()
    
    success = setup_custom_flamingo(args.base_dir, args.excel_file, args.images_dir)
    
    if success:
        print("\n✅ Setup completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()