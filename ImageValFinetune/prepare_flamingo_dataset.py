"""
Fixed dataset preparation for Custom Flamingo training
"""
import os
import json
import pandas as pd
import argparse
from PIL import Image
import finetune_config as config

def prepare_flamingo_dataset(excel_file: str, images_dir: str, output_path: str):
    """Prepare dataset in Flamingo format with better error handling"""
    
    print("📊 Preparing Custom Flamingo dataset...")
    print(f"Excel file: {excel_file}")
    print(f"Images dir: {images_dir}")
    
    # Read Excel file
    try:
        df = pd.read_excel(excel_file)
        print(f"Loaded Excel with {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        return False
    
    # Auto-detect column names
    file_name_col = None
    description_col = None
    
    # Try different possible column names
    possible_file_cols = ['File Name', 'FileName', 'file_name', 'filename', 'image_name', 'Image Name', 'Image']
    possible_desc_cols = ['Description', 'description', 'Caption', 'caption', 'Text', 'text', 'Arabic Caption', 'Arabic_Caption']
    
    for col in possible_file_cols:
        if col in df.columns:
            file_name_col = col
            break
    
    for col in possible_desc_cols:
        if col in df.columns:
            description_col = col
            break
    
    if not file_name_col or not description_col:
        print(f"❌ Required columns not found!")
        print(f"Available columns: {list(df.columns)}")
        print(f"Looking for file column in: {possible_file_cols}")
        print(f"Looking for description column in: {possible_desc_cols}")
        return False
    
    print(f"✅ Using columns: {file_name_col} -> {description_col}")
    
    # Get list of actual image files
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return False
    
    all_files = os.listdir(images_dir)
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    print(f"Found {len(image_files)} image files")
    
    flamingo_data = []
    processed = 0
    skipped = 0
    
    for _, row in df.iterrows():
        if pd.notna(row[file_name_col]) and pd.notna(row[description_col]):
            filename = str(row[file_name_col]).strip()
            description = str(row[description_col]).strip()
            
            # Try to find matching image file
            matching_image = None
            
            # Try exact match first
            if filename in image_files:
                matching_image = filename
            else:
                # Try without extension
                name_without_ext = os.path.splitext(filename)[0]
                matches = [f for f in image_files if f.startswith(name_without_ext)]
                if matches:
                    matching_image = matches[0]
                else:
                    # Try case-insensitive match
                    matches = [f for f in image_files if f.lower().startswith(name_without_ext.lower())]
                    if matches:
                        matching_image = matches[0]
            
            if matching_image:
                image_path = os.path.join(images_dir, matching_image)
                
                # Verify image can be opened
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                    
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
                                "value": description
                            }
                        ]
                    }
                    
                    flamingo_data.append(entry)
                    processed += 1
                    
                except Exception as e:
                    print(f"⚠️ Skipping corrupted image {image_path}: {e}")
                    skipped += 1
            else:
                skipped += 1
                if skipped <= 5:  # Only print first 5 missing files
                    print(f"⚠️ Image not found for: {filename}")
    
    print(f"\n📊 Processing summary:")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    
    if processed == 0:
        print("❌ No valid samples created!")
        return False
    
    # Save dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(flamingo_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Dataset prepared: {len(flamingo_data)} samples saved to {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Prepare Custom Flamingo dataset")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory")
    parser.add_argument("--excel_file", type=str, help="Excel file path")
    parser.add_argument("--images_dir", type=str, help="Images directory path")
    
    args = parser.parse_args()
    
    # Use provided paths or defaults
    excel_file = args.excel_file or "/kaggle/input/trainsubtask2/TrainSubtask2.xlsx"
    images_dir = args.images_dir or "/kaggle/working/Train/images"
    output_path = os.path.join(args.base_dir, "arabic_flamingo_dataset.json")
    
    success = prepare_flamingo_dataset(excel_file, images_dir, output_path)
    
    if success:
        print("✅ Dataset preparation completed successfully!")
    else:
        print("❌ Dataset preparation failed!")
        exit(1)

if __name__ == "__main__":
    main()