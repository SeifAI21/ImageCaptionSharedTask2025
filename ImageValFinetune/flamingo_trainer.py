"""
Custom Flamingo Trainer with AraGPT2-Mega
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW  
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import Optional, List, Dict
from arabic_flamingo_model import ArabicFlamingoModel, apply_cross_attention_patch

class FlamingoDataset(torch.utils.data.Dataset):
    """Dataset for Flamingo training"""
    
    def __init__(self, data_path: str, tokenizer, image_processor):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.image_processor = image_processor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_path = item['images'][0]
        image = Image.open(image_path).convert('RGB')
        image_inputs = self.image_processor(images=image, return_tensors="pt")
        
        # Process conversation
        conversation = item['conversations']
        prompt = conversation[0]['value']  # Human message with <image>
        response = conversation[1]['value']  # GPT response
        
        # Create full text
        full_text = f"{prompt} {response}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # Create labels (mask prompt part)
        prompt_encoding = self.tokenizer(prompt, add_special_tokens=False)
        prompt_length = len(prompt_encoding['input_ids'])
        
        labels = encoding['input_ids'].clone()
        labels[0, :prompt_length] = -100  # Ignore prompt in loss
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'images': image_inputs['pixel_values'].squeeze()
        }

class ArabicFlamingoTrainer:
    """Trainer for Arabic Flamingo model"""
    
    def __init__(
        self,
        base_dir: str = "/kaggle/working",  # FIXED: Updated default path
        model_name: str = "aubmindlab/aragpt2-mega"
    ):
        self.base_dir = base_dir
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔄 Using device: {self.device}")
        
        # Apply cross-attention patch
        apply_cross_attention_patch()
        
        # Initialize model
        print("🔄 Initializing Arabic Flamingo model...")
        self.model = ArabicFlamingoModel(lang_model_path=model_name)
        self.tokenizer = self.model.tokenizer
        self.image_processor = self.model.image_processor
        
        # Move to device
        self.model = self.model.to(self.device)
        
        print("✅ Arabic Flamingo Trainer initialized")
    
    def setup_lora(self, lora_rank: int = 8, lora_alpha: int = 16):
        """Setup LoRA for efficient fine-tuning"""
        print("🔧 Setting up LoRA...")
        
        # LoRA config for AraGPT2 language model
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["c_attn", "c_proj", "c_fc"],  # AraGPT2 modules
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA to language model
        self.model.lang_model = get_peft_model(self.model.lang_model, lora_config)
        
        # Only train LoRA parameters + cross-attention + perceiver
        for name, param in self.model.named_parameters():
            if any(target in name for target in ["lora", "cross_attn", "perceiver", "vision_projection"]):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    def train(
        self,
        dataset_path: str,
        num_epochs: int = 3,
        batch_size: int = 1,
        learning_rate: float = 5e-6,
        output_dir: str = None
    ):
        """Train the Arabic Flamingo model"""
        
        output_dir = output_dir or os.path.join(self.base_dir, "arabic_flamingo_model")
        os.makedirs(output_dir, exist_ok=True)
        
        print("🚀 Starting Arabic Flamingo training...")
        print(f"📊 Dataset: {dataset_path}")
        print(f"📊 Output: {output_dir}")
        print(f"📊 Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        
        # Setup LoRA
        self.setup_lora()
        
        # Create dataset
        print("📚 Loading dataset...")
        dataset = FlamingoDataset(dataset_path, self.tokenizer, self.image_processor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"📚 Dataset loaded: {len(dataset)} samples")
        
        # Setup optimizer and scheduler
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=0.01
        )
        
        num_training_steps = len(dataloader) * num_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_training_steps // 10,
            num_training_steps=num_training_steps
        )
        
        print(f"🔧 Optimizer: AdamW with {len(trainable_params)} trainable parameter groups")
        print(f"🔧 Scheduler: Cosine with {num_training_steps} total steps")
        
        # Training loop
        self.model.train()
        total_loss = 0
        
        for epoch in range(num_epochs):
            print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
            
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        images=batch['images'],
                        labels=batch['labels']
                    )
                    
                    loss = outputs.loss
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    total_loss += loss.item()
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'avg_loss': f"{epoch_loss / (batch_idx + 1):.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                    })
                    
                    # Save checkpoint every 50 steps
                    if (batch_idx + 1) % 50 == 0:
                        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{epoch}-{batch_idx}")
                        self.save_model(checkpoint_dir)
                        print(f"💾 Checkpoint saved: step {batch_idx + 1}")
                
                except Exception as e:
                    print(f"❌ Error in batch {batch_idx}: {e}")
                    continue
            
            # Save epoch checkpoint
            epoch_dir = os.path.join(output_dir, f"epoch-{epoch + 1}")
            self.save_model(epoch_dir)
            
            print(f"✅ Epoch {epoch + 1} completed. Average loss: {epoch_loss / len(dataloader):.4f}")
        
        # Save final model
        final_dir = os.path.join(output_dir, "final_model")
        self.save_model(final_dir)
        
        print(f"🎉 Training completed! Model saved to: {final_dir}")
    
    def save_model(self, save_dir: str):
        """Save the trained model"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the full model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'model_name': self.model_name,
                'base_dir': self.base_dir
            }
        }, os.path.join(save_dir, 'arabic_flamingo_model.pt'))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)
        
        print(f"💾 Model saved to: {save_dir}")
    
    def load_model(self, load_dir: str):
        """Load a trained model"""
        checkpoint_path = os.path.join(load_dir, 'arabic_flamingo_model.pt')
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded from: {load_dir}")
        else:
            print(f"❌ Checkpoint not found: {checkpoint_path}")
    
    def generate_caption(self, image_path: str, prompt: str = "وصف هذه الصورة:") -> str:
        """Generate Arabic caption for an image"""
        self.model.eval()
        
        with torch.no_grad():
            caption = self.model.generate_caption(
                image_path=image_path,
                prompt=prompt,
                max_new_tokens=100,
                temperature=0.7
            )
        
        return caption