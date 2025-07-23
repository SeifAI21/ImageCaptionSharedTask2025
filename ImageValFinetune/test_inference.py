"""
Test inference with trained Flamingo model
"""
import os
import torch
from PIL import Image
import argparse
from arabic_flamingo_model import ArabicFlamingoModel, apply_cross_attention_patch
import finetune_config as config

class FlamingoInference:
    """Simple inference class for trained Flamingo model"""
    
    def __init__(self, checkpoint_path: str):
        """Initialize inference with trained checkpoint"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔄 Loading Flamingo model from: {checkpoint_path}")
        print(f"Device: {self.device}")
        
        # Apply cross-attention patch
        apply_cross_attention_patch()
        
        # Initialize model
        self.model = ArabicFlamingoModel(
            lang_model_path=config.DEFAULT_MODEL_NAME,
            vision_encoder_path=config.VISION_MODEL_NAME
        )
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        
        # Set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model loaded and ready for inference!")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load trained weights"""
        if os.path.isdir(checkpoint_path):
            # Look for .pt files in directory
            pt_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.pt')]
            if pt_files:
                checkpoint_file = os.path.join(checkpoint_path, pt_files[0])
            else:
                raise FileNotFoundError(f"No .pt files found in {checkpoint_path}")
        else:
            checkpoint_file = checkpoint_path
        
        print(f"Loading weights from: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
    
    def generate_caption(self, image_path: str, prompt: str = "وصف هذه الصورة:") -> str:
        """Generate Arabic caption for an image"""
        if not os.path.exists(image_path):
            return f"❌ Image not found: {image_path}"
        
        try:
            with torch.no_grad():
                caption = self.model.generate_caption(
                    image_path=image_path,
                    prompt=prompt,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True
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
        caption = self.generate_caption(image_path)
        print(f"📝 Generated Caption: {caption}")
        
        return caption
    
    def test_multiple_images(self, image_dir: str, max_images: int = 5):
        """Test inference on multiple images"""
        print(f"\n🖼️  Testing multiple images from: {image_dir}")
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
    
    def interactive_test(self):
        """Interactive testing mode"""
        print("\n🎮 Interactive Testing Mode")
        print("Enter image path (or 'quit' to exit):")
        
        while True:
            image_path = input("\nImage path: ").strip()
            
            if image_path.lower() in ['quit', 'exit', 'q']:
                break
            
            if not image_path:
                continue
            
            self.test_single_image(image_path)


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Test Flamingo model inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--image_dir", type=str, help="Directory with multiple images")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--max_images", type=int, default=5, help="Max images to test")
    
    args = parser.parse_args()
    
    # Initialize inference
    try:
        inferencer = FlamingoInference(args.checkpoint)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Run tests
    if args.interactive:
        inferencer.interactive_test()
    elif args.image:
        inferencer.test_single_image(args.image)
    elif args.image_dir:
        inferencer.test_multiple_images(args.image_dir, args.max_images)
    else:
        print("Please specify --image, --image_dir, or --interactive")


if __name__ == "__main__":
    main()