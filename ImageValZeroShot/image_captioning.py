#!/usr/bin/env python3
import os
import csv
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


class ArabicImageCaptioner:
    """Generate Arabic captions for images using Gemma model (manual, no pipeline)."""

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

        # Add HF token support like the working example
        kwargs = {}
        # You can add HF_TOKEN here if needed
        # kwargs["token"] = "your_hf_token"

        self.model = AutoModelForImageTextToText.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            **kwargs
        )
        self.processor = AutoProcessor.from_pretrained(
            path,
            trust_remote_code=True,
            **kwargs
        )
        print("Model and processor loaded successfully!\n")

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image to meet Gemma3n requirements (512x512)"""
        if image.mode != "RGB":
            image = image.convert("RGB")

        target_size = (512, 512)
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if aspect_ratio > 1:
            new_width = target_size[0]
            new_height = int(target_size[0] / aspect_ratio)
        else:
            new_height = target_size[1]
            new_width = int(target_size[1] * aspect_ratio)

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        processed_image = Image.new("RGB", target_size, (255, 255, 255))

        x_offset = (target_size[0] - new_width) // 2
        y_offset = (target_size[1] - new_height) // 2
        processed_image.paste(image, (x_offset, y_offset))

        return processed_image

    def generate_caption(self, image_path, max_new_tokens=128):
        """Generate Arabic caption for a single image using chat template."""
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Preprocess image to meet Gemma3n requirements
            processed_image = self.preprocess_image(image)

            # Use chat template format like the working example
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "أنت خبير في فهم المشاهد البصرية وإنشاء التعليقات متعددة اللغات. تخصصك هو تحليل الصور التاريخية المتعلقة بالنكبة الفلسطينية والاحتلال الإسرائيلي."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": processed_image},
                        {
                            "type": "text",
                            "text": "حلل محتوى هذه الصورة وقدم تسمية موجزة وذات معنى باللغة العربية (15–50 كلمة) تعكس محتوى المشهد والسياق العاطفي، وبصياغة طبيعية مناسبة ثقافيًا. التعليق يجب أن يكون باللغة العربية فقط."
                        }
                    ]
                }
            ]

            # Apply chat template and tokenize
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device, dtype=self.model.dtype)
            
            input_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    disable_compile=True,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )

            # Decode only the generated part
            response = self.processor.batch_decode(
                outputs[:, input_len:],
                skip_special_tokens=True,
            )[0].strip()

            return response

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return ""




    def process_folder(self, image_folder, output_csv, supported_formats=('.png', '.jpg', '.jpeg')):
        if not os.path.isdir(image_folder):
            raise FileNotFoundError(f"Image folder not found: {image_folder}")

        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(supported_formats)]
        if not image_files:
            print(f"No supported images in {image_folder}")
            return

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8') as out:
            writer = csv.writer(out)
            writer.writerow(["image_file", "arabic_caption"])

            for img in tqdm(image_files, desc="Processing images"):
                path = os.path.join(image_folder, img)
                cap = self.generate_caption(path)
                writer.writerow([img, cap if cap else ""])

        print(f"Done! Captions saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="google/gemma-3n-E4B-it")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args = parser.parse_args()

    captioner = ArabicImageCaptioner(model_name=args.model_name, checkpoint_path=args.checkpoint_path)
    captioner.load_model()
    captioner.process_folder(args.image_folder, args.output_csv)


if __name__ == "__main__":
    main()
