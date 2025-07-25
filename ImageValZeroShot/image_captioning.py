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
from transformers import AutoModelForImageTextToText, AutoProcessor


class ArabicImageCaptioner:
    """Class for generating Arabic captions for images using Gemma model."""
    
    def __init__(self, model_name="google/gemma-3n-E4B-it", checkpoint_path=None):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load the model and processor."""
        print(f"Loading model: {self.model_name} on {self.device}")
        path = self.checkpoint_path or self.model_name

        self.model = AutoModelForImageTextToText.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            use_auth_token=True
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_auth_token=True
        )
        print("Model and processor loaded successfully!\n")

    def generate_caption(self, image_path, max_new_tokens=128):
        """Generate Arabic caption for a single image."""
        try:
            image = Image.open(image_path).convert("RGB")
            
            prompt = (
                "أنت خبير في فهم المشاهد البصرية وإنشاء التعليقات متعددة اللغات. "
                "حلل محتوى هذه الصورة المتعلقة بالنكبة الفلسطينية أو الاحتلال الإسرائيلي، "
                "وقدم تسمية موجزة وذات معنى باللغة العربية (15–50 كلمة)، "
                "تعكس محتوى المشهد والسياق العاطفي، وبصياغة طبيعية مناسبة ثقافيًا."
            )
            
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            caption = self.processor.decode(
                generated_ids[0],
                skip_special_tokens=True
            ).strip()
            return caption
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return ""

    def process_folder(self, image_folder, output_csv, supported_formats=('.png', '.jpg', '.jpeg')):
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
                if cap:
                    writer.writerow([img, cap])
                else:
                    writer.writerow([img, ""])
        
        print(f"\nDone! Captions saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Arabic captions for images using Gemma model"
    )
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_csv",   type=str, required=True)
    parser.add_argument("--model_name",   type=str, default="google/gemma-3n-E4B-it")
    parser.add_argument("--max_tokens",   type=int, default=128)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args = parser.parse_args()
    
    captioner = ArabicImageCaptioner(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path
    )
    captioner.load_model()
    captioner.process_folder(
        image_folder=args.image_folder,
        output_csv=args.output_csv
    )


if __name__ == "__main__":
    main()
