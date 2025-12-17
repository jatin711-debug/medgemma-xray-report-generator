"""
Debug script to verify MedGemma attention tensor shapes and image token positions.
Run this BEFORE relying on the heatmap code to ensure assumptions are correct.
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
from io import BytesIO

def debug_medgemma_attention():
    """Load MedGemma and inspect attention tensor shapes."""
    
    model_id = "google/medgemma-4b-it"
    print(f"Loading {model_id}...")
    
    # Load model and processor
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # Required for attention extraction
        device_map="auto"
    )
    
    # Download a sample chest X-ray (with proper headers to avoid 403)
    print("Downloading sample image...")
    url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print(f"  Downloaded image: {image.size}")
    except Exception as e:
        print(f"  Failed to download ({e}), creating synthetic test image...")
        # Create a simple grayscale test image (simulating chest X-ray)
        import numpy as np
        # Create gradient image to simulate chest X-ray
        arr = np.zeros((512, 512), dtype=np.uint8)
        # Add some structure
        arr[100:400, 150:350] = 180  # Lung fields
        arr[200:350, 220:290] = 100  # Heart shadow
        image = Image.fromarray(arr).convert("RGB")
        print(f"  Created synthetic test image: {image.size}")
    
    # Resize to expected size
    image_896 = image.resize((896, 896))
    image_224 = image.resize((224, 224))  # Also test the "wrong" size
    
    # Prepare input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_896},
                {"type": "text", "text": "Describe any abnormalities in this chest X-ray."}
            ]
        }
    ]
    
    print("\n" + "="*60)
    print("PROCESSOR CONFIG:")
    print("="*60)
    
    # Check processor config
    if hasattr(processor, 'image_processor'):
        ip = processor.image_processor
        print(f"  image_processor type: {type(ip).__name__}")
        if hasattr(ip, 'size'):
            print(f"  size: {ip.size}")
        if hasattr(ip, 'patch_size'):
            print(f"  patch_size: {ip.patch_size}")
        if hasattr(ip, 'image_mean'):
            print(f"  image_mean: {ip.image_mean}")
    else:
        print("  No image_processor attribute found!")
    
    # Prepare inputs
    print("\nPreparing inputs...")
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    
    print("\n" + "="*60)
    print("INPUT TENSOR SHAPES:")
    print("="*60)
    for key, val in inputs.items():
        if hasattr(val, 'shape'):
            print(f"  {key}: {val.shape}")
        else:
            print(f"  {key}: {type(val)}")
    
    input_len = inputs["input_ids"].shape[-1]
    print(f"\n  Total sequence length: {input_len}")
    
    # Run forward pass with attention
    print("\nRunning forward pass with output_attentions=True...")
    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_attentions=True,
            use_cache=False
        )
    
    print("\n" + "="*60)
    print("OUTPUT ATTRIBUTES:")
    print("="*60)
    for attr in dir(outputs):
        if not attr.startswith('_'):
            val = getattr(outputs, attr)
            if val is not None:
                if hasattr(val, '__len__') and not isinstance(val, str):
                    print(f"  {attr}: len={len(val)}")
                elif hasattr(val, 'shape'):
                    print(f"  {attr}: shape={val.shape}")
                else:
                    print(f"  {attr}: {type(val).__name__}")
    
    # Check attentions
    print("\n" + "="*60)
    print("ATTENTION ANALYSIS:")
    print("="*60)
    
    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        attentions = outputs.attentions
        print(f"  Number of layers: {len(attentions)}")
        print(f"  First layer tensor shape: {attentions[0].shape}")
        print(f"  Last layer tensor shape: {attentions[-1].shape}")
        
        # Expected shape: (batch, num_heads, seq_len, seq_len)
        batch, num_heads, seq_len_q, seq_len_k = attentions[0].shape
        print(f"\n  Interpreted as:")
        print(f"    batch_size: {batch}")
        print(f"    num_heads: {num_heads}")
        print(f"    query_seq_len: {seq_len_q}")
        print(f"    key_seq_len: {seq_len_k}")
        
        # Check for image tokens
        print("\n" + "="*60)
        print("IMAGE TOKEN POSITION ANALYSIS:")
        print("="*60)
        
        # If 256 image tokens exist, they should show distinctive attention patterns
        # Try to detect them by analyzing attention variance
        last_layer_attn = attentions[-1][0].float()  # (heads, seq, seq)
        avg_attn = last_layer_attn.mean(dim=0)  # (seq, seq)
        
        # Compute self-attention strength along diagonal
        diag = avg_attn.diagonal()
        print(f"  Diagonal attention (self-attention) stats:")
        print(f"    mean: {diag.mean():.4f}")
        print(f"    std: {diag.std():.4f}")
        print(f"    min: {diag.min():.4f}")
        print(f"    max: {diag.max():.4f}")
        
        # Try to identify image token block
        # Image tokens often have similar attention patterns
        print(f"\n  Sequence regions (assuming 256 image tokens):")
        if seq_len_k > 256:
            first_256 = avg_attn[:, :256].mean().item()
            last_256 = avg_attn[:, -256:].mean().item()
            middle = avg_attn[:, 128:384].mean().item() if seq_len_k > 384 else 0
            
            print(f"    First 256 tokens avg attention: {first_256:.4f}")
            print(f"    Last 256 tokens avg attention: {last_256:.4f}")
            if seq_len_k > 384:
                print(f"    Middle tokens (128-384) avg attention: {middle:.4f}")
            
            print(f"\n  >>> If image tokens are at START, first_256 should be higher")
            print(f"  >>> If image tokens are at END, last_256 should be higher")
    else:
        print("  ❌ NO ATTENTIONS FOUND IN OUTPUT!")
        print("  This means output_attentions=True is not working as expected.")
        print("  The heatmap code will NOT work with this model.")
    
    print("\n" + "="*60)
    print("CONCLUSIONS:")
    print("="*60)
    
    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        attn_shape = attentions[0].shape
        print(f"  ✅ Attentions ARE available with shape: {attn_shape}")
        print(f"  ✅ Should be able to extract heatmaps")
        print(f"\n  RECOMMENDATIONS:")
        print(f"    1. Image tokens likely at positions 0-255 (start) or seq_len-256 (end)")
        print(f"    2. Use layer averaging for stable heatmaps")
        print(f"    3. Grid should be sqrt(num_image_tokens) = 16x16 for 256 tokens")
    else:
        print(f"  ❌ Attentions NOT available")
        print(f"  ❌ Need alternative approach (zone-based or gradient-based)")

if __name__ == "__main__":
    debug_medgemma_attention()
