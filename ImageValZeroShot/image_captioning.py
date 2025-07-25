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
from transformers import AutoModelForImageTextToText, AutoProcessor  # Updated import


class ArabicImageCaptioner:
    """Class for generating Arabic captions for images using Gemma model."""
    
    def __init__(self, model_name="google/gemma-3n-E4B-it", checkpoint_path=None):
        """
        Initialize the captioner with the specified model.
        
        Args:
            model_name (str): Name of the Gemma model to use
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        
    def load_model(self):
        """Load the model and processor."""
        print(f"Loading model: {self.model_name}")
        print(f"Using device: {self.device}")
        
        loading_path = self.checkpoint_path if self.checkpoint_path else self.model_name

        self.model = AutoModelForImageTextToText.from_pretrained(  # Updated class
            loading_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
                
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        print("Model and processor loaded successfully!")
        
    def generate_caption(self, image_path, max_new_tokens=128):
        """
        Generate Arabic caption for a single image.
        
        Args:
            image_path (str): Path to the image file
            max_new_tokens (int): Maximum number of tokens to generate
            
        Returns:
            str: Generated Arabic caption
        """
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Simplified prompt for Gemma (no chat template)
            prompt = (
                "You are an expert in visual scene understanding and multilingual caption generation. "
                "Analyze the content of this image, which is potentially related to the Palestinian Nakba "
                "and Israeli occupation of Palestine, and provide a concise and meaningful caption in Arabic - about 15 to 50 words. "
                "The caption should reflect the scene's content, emotional context, and should be natural and culturally appropriate. "
                "Do not include any English or metadata — The caption must be in Arabic."
            )
            
            # Process inputs for Gemma
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            # Decode and clean output
            output_text = self.processor.decode(
                generated_ids[0], 
                skip_special_tokens=True
            )
            
            # Remove the original prompt from output
            arabic_caption = output_text.replace(prompt, "").strip()
            
            return arabic_caption
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return ""
    
    # ... rest of the methods remain the same ...
    def process_folder(self, image_folder, output_csv, supported_formats=('.png', '.jpg', '.jpeg')):
        """
        Process all images in a folder and save captions to CSV.
        
        Args:
            image_folder (str): Path to folder containing images
            output_csv (str): Path to output CSV file
            supported_formats (tuple): Supported image file extensions
        """
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Image folder not found: {image_folder}")
        
        # Get list of image files
        image_files = [
            f for f in os.listdir(image_folder) 
            if f.lower().endswith(supported_formats)
        ]
        
        if not image_files:
            print(f"No supported image files found in {image_folder}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        # Process images and write to CSV
        with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["image_file", "arabic_caption"])
            
            for image_file in tqdm(image_files, desc="Processing images"):
                image_path = os.path.join(image_folder, image_file)
                caption = self.generate_caption(image_path)
                
                if caption:
                    print(f"{image_file}: {caption}")
                    writer.writerow([image_file, caption])
                else:
                    print(f"Failed to generate caption for {image_file}")
        
        print(f"Processing complete! Results saved to: {output_csv}")


def main():
    """Main function to handle command line arguments and run the captioning process."""
    parser = argparse.ArgumentParser(
        description="Generate Arabic captions for images using Gemma model"
    )
    parser.add_argument(
        "--image_folder", 
        type=str, 
        required=True,
        help="Path to folder containing images"
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        required=True,
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="google/gemma-3n-E4B-it",  # Changed default
        help="Model name to use (default: google/gemma-3n-E4B-it)"
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=128,
        help="Maximum number of tokens to generate (default: 128)"
    )
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        default=None,
        help="Path to checkpoint file"
    )
    
    args = parser.parse_args()
    
    # Initialize captioner
    captioner = ArabicImageCaptioner(model_name=args.model_name, checkpoint_path=args.checkpoint_path)
    captioner.load_model()
    
    # Process images
    captioner.process_folder(
        image_folder=args.image_folder,
        output_csv=args.output_csv
    )


if __name__ == "__main__":
    main()