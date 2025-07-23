"""
Test inference with base Flamingo model (no fine-tuning checkpoint needed)
"""
import os
import torch
from PIL import Image
import argparse
from arabic_flamingo_model import ArabicFlamingoModel, apply_cross_attention_patch
import finetune_config as config

class BaseFlamingoInference:
    """Inference class for base Flamingo model (no checkpoint required)"""
    
    def __init__(self, model_name: str = None):
        """Initialize inference with base pre-trained model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use provided model or default
        self.model_name = model_name or config.DEFAULT_MODEL_NAME
        
        print(f"🔄 Loading BASE Flamingo model...")
        print(f"Language Model: {self.model_name}")
        print(f"Vision Model: {config.VISION_MODEL_NAME}")
        print(f"Device: {self.device}")
        
        # Apply cross-attention patch
        apply_cross_attention_patch()
        
        # Initialize model with base weights only
        self.model = ArabicFlamingoModel(
            lang_model_path=self.model_name,
            vision_encoder_path=config.VISION_MODEL_NAME
        )
        
        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("✅ Base model loaded and ready for inference!")
        print("📝 Note: This is the base model without fine-tuning")
    
    def generate_caption(self, image_path: str, prompt: str = "وصف هذه الصورة:") -> str:
        """Generate Arabic caption for an image using base model"""
        if not os.path.exists(image_path):
            return f"❌ Image not found: {image_path}"
        
        try:
            with torch.no_grad():
                caption = self.model.generate_caption(
                    image_path=image_path,
                    prompt=prompt,
                    max_new_tokens=50,      # Shorter for base model
                    temperature=0.8,        # Slightly higher temperature
                    do_sample=True,
                    top_p=0.9
                )
            return caption.strip()
        except Exception as e:
            return f"❌ Error generating caption: {e}"
    
    def test_single_image(self, image_path: str, show_image: bool = True):
        """Test inference on a single image"""
        print(f"\n🖼️  Testing image: {os.path.basename(image_path)}")
        print("-" * 50)
        
        if show_image:
            try:
                from IPython.display import Image as IPImage, display
                display(IPImage(image_path, width=300))
            except:
                print("(Image display not available)")
        
        # Generate caption
        print("🔄 Generating caption with BASE model...")
        caption = self.generate_caption(image_path)
        print(f"📝 Generated Caption: {caption}")
        
        return caption
    
    def test_multiple_images(self, image_dir: str, max_images: int = 3):
        """Test inference on multiple images"""
        print(f"\n🖼️  Testing multiple images from: {image_dir}")
        print("📝 Using BASE model (no fine-tuning)")
        print("=" * 60)
        
        # Get image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])
        
        image_files = image_files[:max_images]
        
        results = []
        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(image_dir, image_file)
            print(f"\n{i}. {image_file}")
            caption = self.generate_caption(image_path)
            print(f"   Caption: {caption}")
            results.append({"image": image_file, "caption": caption})
        
        return results
    
    def compare_model_sizes(self):
        """Compare different AraGPT2 model sizes"""
        models_to_test = [
            "aubmindlab/aragpt2-medium",
            "aubmindlab/aragpt2-large", 
            "aubmindlab/aragpt2-mega"
        ]
        
        print("\n🔍 Comparing different AraGPT2 model sizes:")
        print("=" * 60)
        
        # Find a test image
        test_images_dir = "/kaggle/working/Train/images"
        if os.path.exists(test_images_dir):
            image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.png'))]
            if image_files:
                test_image = os.path.join(test_images_dir, image_files[0])
                print(f"🖼️  Test image: {image_files[0]}")
                
                for model_name in models_to_test:
                    try:
                        print(f"\n📝 Testing {model_name.split('/')[-1].upper()}:")
                        
                        # Initialize model
                        temp_inferencer = BaseFlamingoInference(model_name)
                        
                        # Generate caption
                        caption = temp_inferencer.generate_caption(test_image)
                        print(f"   Result: {caption}")
                        
                        # Clean up
                        del temp_inferencer
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"   ❌ Error with {model_name}: {e}")
            else:
                print("❌ No test images found")
        else:
            print(f"❌ Test images directory not found: {test_images_dir}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Test BASE Flamingo model inference (no checkpoint)")
    parser.add_argument("--model", type=str, help="Model name (default: config default)")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--image_dir", type=str, help="Directory with multiple images")
    parser.add_argument("--compare_models", action="store_true", help="Compare different model sizes")
    parser.add_argument("--max_images", type=int, default=3, help="Max images to test")
    
    args = parser.parse_args()
    
    # Initialize base model inference
    try:
        inferencer = BaseFlamingoInference(args.model)
    except Exception as e:
        print(f"❌ Failed to load base model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run tests
    if args.compare_models:
        inferencer.compare_model_sizes()
    elif args.image:
        inferencer.test_single_image(args.image)
    elif args.image_dir:
        inferencer.test_multiple_images(args.image_dir, args.max_images)
    else:
        print("Please specify --image, --image_dir, or --compare_models")
        print("\nExample usage:")
        print("  python test_base_inference.py --image_dir /kaggle/working/Train/images")
        print("  python test_base_inference.py --compare_models")


if __name__ == "__main__":
    main()