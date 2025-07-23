"""
Custom Flamingo implementation with AraGPT2-Mega as language backbone
"""
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    CLIPVisionModel, CLIPImageProcessor,
    AutoConfig
)
from PIL import Image
import numpy as np
from typing import Optional, List, Dict, Tuple

class PerceiverResampler(nn.Module):
    """Perceiver Resampler for Flamingo architecture"""
    
    def __init__(
        self,
        dim: int,
        depth: int = 6,
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 64,
        num_media_embeds: int = 257
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, batch_first=True),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                )
            ]))
    
    def forward(self, x):
        batch_size = x.shape[0]
        latents = self.latents.unsqueeze(0).repeat(batch_size, 1, 1)
        
        for ln1, attn, ln2, ff in self.layers:
            # Cross attention
            latents_norm = ln1(latents)
            attended, _ = attn(latents_norm, x, x)
            latents = latents + attended
            
            # Feed forward
            latents = latents + ff(ln2(latents))
        
        return latents

class GatedCrossAttentionBlock(nn.Module):
    """Gated Cross Attention Block for Flamingo"""
    
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.tanh_gate = nn.Tanh()
        self.alpha = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, media_features):
        if media_features is None:
            return x
        
        # Cross attention
        normed_x = self.norm(x)
        attended, _ = self.cross_attn(normed_x, media_features, media_features)
        
        # Gated residual
        gate = self.tanh_gate(self.alpha)
        return x + gate * attended

class ArabicFlamingoModel(nn.Module):
    """Flamingo model with AraGPT2-Mega as language backbone"""
    
    def __init__(
        self,
        vision_encoder_path: str = "openai/clip-vit-large-patch14",
        lang_model_path: str = "aubmindlab/aragpt2-mega",
        cross_attn_every_n_layers: int = 4,
        perceiver_num_latents: int = 64,
        perceiver_depth: int = 6
    ):
        super().__init__()
        
        print(f"🔄 Loading AraGPT2 language model: {lang_model_path}")
        
        # Load AraGPT2 tokenizer and model with trust_remote_code=True
        self.tokenizer = AutoTokenizer.from_pretrained(
            lang_model_path,
            trust_remote_code=True  # FIXED: Added trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.lang_model = AutoModelForCausalLM.from_pretrained(
            lang_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True  # FIXED: Added trust_remote_code=True
        )
        
        print(f"🔄 Loading CLIP vision encoder: {vision_encoder_path}")
        
        # Load CLIP vision encoder
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_encoder_path)
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_encoder_path)
        
        # Freeze vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # Get dimensions
        vision_dim = self.vision_encoder.config.hidden_size  # 1024 for CLIP-Large
        lang_dim = self.lang_model.config.n_embd  # 1536 for AraGPT2-mega
        
        print(f"📐 Vision dim: {vision_dim}, Language dim: {lang_dim}")
        
        # Perceiver resampler (Flamingo's key component)
        self.perceiver = PerceiverResampler(
            dim=lang_dim,
            depth=perceiver_depth,
            num_latents=perceiver_num_latents,
            num_media_embeds=257  # CLIP outputs 257 tokens
        )
        
        # Vision-to-language projection
        self.vision_projection = nn.Linear(vision_dim, lang_dim)
        
        # Add cross-attention layers to AraGPT2
        self._add_cross_attention_layers(cross_attn_every_n_layers)
        
        # Special tokens
        self.media_token_id = self.tokenizer.encode("<image>", add_special_tokens=False)[0] if "<image>" in self.tokenizer.get_vocab() else self.tokenizer.eos_token_id
        
        print("✅ Arabic Flamingo model initialized successfully!")
    
    def _add_cross_attention_layers(self, every_n_layers: int):
        """Add cross-attention layers to AraGPT2 transformer blocks"""
        print(f"🔧 Adding cross-attention every {every_n_layers} layers")
        
        # Get AraGPT2 transformer blocks
        transformer_blocks = self.lang_model.transformer.h
        lang_dim = self.lang_model.config.n_embd
        
        # Add cross-attention to every N-th layer
        for i in range(0, len(transformer_blocks), every_n_layers):
            if i < len(transformer_blocks):
                block = transformer_blocks[i]
                
                # Add cross-attention module
                block.cross_attn = GatedCrossAttentionBlock(
                    dim=lang_dim,
                    heads=8
                )
                
                print(f"  ✓ Added cross-attention to layer {i}")
    
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using CLIP and Perceiver"""
        with torch.no_grad():
            # Get CLIP vision features
            vision_outputs = self.vision_encoder(pixel_values=images)
            image_features = vision_outputs.last_hidden_state  # [batch, 257, 1024]
        
        # Project to language model dimension
        image_features = self.vision_projection(image_features)  # [batch, 257, 1536]
        
        # Apply Perceiver resampler
        resampled_features = self.perceiver(image_features)  # [batch, 64, 1536]
        
        return resampled_features
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """Forward pass with cross-attention"""
        
        # Encode images if provided
        media_features = None
        if images is not None:
            media_features = self.encode_images(images)
        
        # Store media features for cross-attention layers
        self._media_features = media_features
        
        # Forward through AraGPT2 with cross-attention
        outputs = self.lang_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False
        )
        
        return outputs
    
# Replace the generate_caption method (around line 200) with this:

def generate_caption(
    self, 
    image_path: str, 
    prompt: str = "وصف هذه الصورة:",
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    do_sample: bool = True
) -> str:
    """Generate Arabic caption for an image"""
    
    # Load and preprocess image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path
    
    # FIXED: Process image correctly
    inputs = self.image_processor(images=image, return_tensors="pt")
    
    # FIXED: Handle the dict properly
    if isinstance(inputs, dict) and 'pixel_values' in inputs:
        pixel_values = inputs['pixel_values']
    else:
        raise ValueError(f"Expected dict with 'pixel_values', got: {type(inputs)}")
    
    if torch.cuda.is_available():
        pixel_values = pixel_values.cuda()
    
    # FIXED: Encode image with correct tensor
    inputs = self.image_processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values']  # Handle as dict
    media_features = self.encode_images(pixel_values)
    
    # Prepare text prompt
    full_prompt = f"<image> {prompt}"
    prompt_tokens = self.tokenizer(
        full_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    if torch.cuda.is_available():
        prompt_tokens = {k: v.cuda() for k, v in prompt_tokens.items()}
    
    # Generate caption
    with torch.no_grad():
        outputs = self.lang_model.generate(
            input_ids=prompt_tokens["input_ids"],
            attention_mask=prompt_tokens["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
    
    # Decode generated text
    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part
    if prompt in generated_text:
        caption = generated_text.split(prompt)[-1].strip()
    else:
        caption = generated_text.strip()
    
    return caption

def apply_cross_attention_patch():
    """Apply cross-attention patch to AraGPT2 blocks"""
    try:
        # Try to import AraGPT2 specific modules
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block
        
        def patched_forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
            """Patched forward method for AraGPT2 blocks with cross-attention"""
            
            # Store original forward for fallback
            if not hasattr(self, '_original_forward'):
                self._original_forward = super(GPT2Block, self).forward
            
            try:
                # Original AraGPT2 block forward
                residual = hidden_states
                hidden_states = self.ln_1(hidden_states)
                attn_outputs = self.attn(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions
                )
                attn_output = attn_outputs[0]
                outputs = attn_outputs[1:]
                
                hidden_states = attn_output + residual
                
                # Apply cross-attention if available and media features exist
                if hasattr(self, 'cross_attn'):
                    # Check for media features in the model
                    model = None
                    for module in self.modules():
                        if hasattr(module, '_media_features'):
                            model = module
                            break
                    
                    if model is not None and hasattr(model, '_media_features') and model._media_features is not None:
                        hidden_states = self.cross_attn(hidden_states, model._media_features)
                
                # Feed forward
                residual = hidden_states
                hidden_states = self.ln_2(hidden_states)
                feed_forward_hidden_states = self.mlp(hidden_states)
                hidden_states = residual + feed_forward_hidden_states
                
                if use_cache:
                    outputs = (hidden_states,) + outputs
                else:
                    outputs = (hidden_states,) + outputs[1:]
                
                return outputs
                
            except Exception as e:
                print(f"⚠️ Cross-attention failed, using original forward: {e}")
                return self._original_forward(hidden_states, layer_past, attention_mask, head_mask, use_cache, output_attentions)
        
        # Apply the patch
        GPT2Block.forward = patched_forward
        print("✅ Cross-attention patch applied successfully")
        
    except ImportError as e:
        print(f"⚠️ Could not patch cross-attention: {e}")
        print("Proceeding without cross-attention patch")