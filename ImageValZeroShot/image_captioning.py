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
        print(f"Loading model: {self.model_name} on {self.device}")
        path = self.checkpoint_path or self.model_name

        self.model = AutoModelForImageTextToText.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            path,
            trust_remote_code=True
        )
        print("Model and processor loaded successfully!\n")

    def generate_caption(self, image_path, max_new_tokens=128):
        try:
            image = Image.open(image_path).convert("RGB")

            # Check Gemma's special image token
            img_token = getattr(self.processor.tokenizer, "image_token", "<image>")
            prompt = (
                f"{img_token} "
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
