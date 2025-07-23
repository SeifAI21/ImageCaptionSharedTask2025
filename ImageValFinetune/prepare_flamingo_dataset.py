"""
Prepare dataset for Custom Flamingo training
"""
import os
import json
import pandas as pd
import argparse
from PIL import Image
import finetune_config as config

def prepare_flamingo_dataset(excel_file: str, images_dir: str, output_path: str):
    """Prepare dataset in Flamingo format"""
    
    print("📊 Preparing Custom Flamingo dataset...")
    
    # Read Excel file
    df = pd.read_excel(excel_file)
    
    flamingo_data = []
    
    for _, row in df.iterrows():
        if pd.notna(row['File Name']) and pd.notna(row['Description']):
            image_path = os.path.join(images_dir, row['File Name'])
            
            if os.path.exists(image_path):
                # Create Flamingo format entry
                entry = {
                    "images": [image_path],
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image> وصف هذه الصورة:"
                        },
                        {
                            "from": "gpt", 
                            "value": str(row['Description'])
                        }
                    ]
                }
                
                flamingo_data.append(entry)
    
    # Save dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(flamingo_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Dataset prepared: {len(flamingo_data)} samples saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare Custom Flamingo dataset")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory")
    parser.add_argument("--excel_file", type=str, help="Excel file path")
    parser.add_argument("--images_dir", type=str, help="Images directory path")
    
    args = parser.parse_args()
    
    # Use provided paths or defaults
    excel_file = args.excel_file or config.DEFAULT_PATHS["train_excel"].replace("/content/drive/MyDrive/ImageVal", args.base_dir)
    images_dir = args.images_dir or config.DEFAULT_PATHS["train_images_dir"].replace("/content/drive/MyDrive/ImageVal", args.base_dir)
    output_path = os.path.join(args.base_dir, config.DATASET_CONFIG["name"] + ".json")
    
    prepare_flamingo_dataset(excel_file, images_dir, output_path)

if __name__ == "__main__":
    main()