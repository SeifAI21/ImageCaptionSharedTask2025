#!/usr/bin/env python3
"""
Arabic Image Captioning using Gemma model
This script processes images and generates Arabic captions for historical content.
"""

import os
import csv
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from transformers import pipeline


class ArabicImageCaptioner:
    """Class for generating Arabic captions for images using Gemma model."""
    
    def __init__(self, model_name="google/gemma-3n-E4B-it"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the pipeline for image-to-text
        print(f"Setting up pipeline with {model_name} on device {self.device}")
        self.pipe = pipeline(
            "image-to-text",
            model=model_name,
            tokenizer=model_name,
            feature_extractor=model_name,
            device=0 if torch.cuda.is_available() else -1,
            trust_remote_code=True
            
        )
        print("Pipeline ready!\n")

    def generate_caption(self, image_path, max_new_tokens=128):
        """Generate an Arabic caption for a single image via the pipeline."""
        try:
            image = Image.open(image_path).convert("RGB")
            outputs = self.pipe(
                image,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            caption = outputs[0]["generated_text"].strip()
            return caption
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return ""

    def process_folder(self, image_folder, output_csv, supported_formats=(".png", ".jpg", ".jpeg")):
        """Process all images in a folder and save captions to CSV."""
        image_folder = image_folder.strip()
        if not os.path.isdir(image_folder):
            raise FileNotFoundError(f"Image folder not found: {image_folder}")
        
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(supported_formats)]
        if not image_files:
            print(f"No supported image files found in {image_folder}")
            return
        
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8') as out:
            writer = csv.writer(out)
            writer.writerow(["image_file", "arabic_caption"])
            
            for img in tqdm(image_files, desc="Processing images"):
                path = os.path.join(image_folder, img)
                cap = self.generate_caption(path)
                writer.writerow([img, cap])
        
        print(f"\nDone! Captions saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Arabic captions for images using Gemma model"
    )
    parser.add_argument(
        "--image_folder", type=str, required=True,
        help="Path to folder containing images"
    )
    parser.add_argument(
        "--output_csv", type=str, required=True,
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--model_name", type=str, default="google/gemma-3n-E4B-it",
        help="Model name to use (default: google/gemma-3n-E4B-it)"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=128,
        help="Maximum number of tokens to generate (default: 128)"
    )
    args = parser.parse_args()
    
    captioner = ArabicImageCaptioner(model_name=args.model_name)
    captioner.process_folder(
        image_folder=args.image_folder,
        output_csv=args.output_csv
    )


if __name__ == "__main__":
    main()
