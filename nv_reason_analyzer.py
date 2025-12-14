"""
NV-Reason-CXR-3B Analyzer Module

NVIDIA's specialized vision-language model for medical reasoning and chest X-ray interpretation.
Based on Qwen2.5-VL-3B architecture with enhanced medical reasoning capabilities.

This module provides:
- Model loading with memory optimization
- Structured reasoning output parsing (<thinking> and <answer> tags)
- Attention-based heatmap generation for abnormality localization
"""

import os
import re
import gc
import logging
from typing import Optional, Any, Tuple
import numpy as np
import cv2

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)


class NVReasonFindings(BaseModel):
    """Structured findings from NV-Reason-CXR-3B model."""
    
    thinking: str = Field(default="", description="Clinical reasoning chain from the model")
    answer: str = Field(default="", description="Final diagnosis/answer")
    abnormalities: list[str] = Field(default_factory=list, description="Detected abnormalities")
    support_devices: list[str] = Field(default_factory=list, description="Detected support devices (tubes, lines, etc.)")
    raw_response: str = Field(default="", description="Complete raw model response")


class NVReasonCXRAnalyzer:
    """
    NVIDIA NV-Reason-CXR-3B analyzer for chest X-ray interpretation.
    
    This class handles:
    - Model loading with GPU optimization
    - Structured inference with thinking/answer parsing
    - Attention-based heatmap generation for abnormality localization
    
    The model is based on Qwen2.5-VL-3B and provides step-by-step medical reasoning.
    """
    
    def __init__(self):
        self.model_id = "nvidia/NV-Reason-CXR-3B"
        self.processor: Optional[Any] = None
        self.model: Optional[Any] = None
        self.model_loaded = False
        
        # Heatmap generation settings
        self._HEATMAP_THRESHOLD_PERCENTILE = 95
        self._GAUSSIAN_BLUR_KERNEL_SIZE = (25, 25)
        self._HEATMAP_OVERLAY_ALPHA = 0.5
    
    def get_system_memory_info(self) -> str:
        """Retrieve GPU/system memory information for monitoring."""
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                return f"GPU Memory: {gpu_memory_gb:.1f}GB total, {allocated:.1f}GB allocated"
            else:
                import psutil
                ram = psutil.virtual_memory()
                return f"RAM: {ram.total / (1024**3):.1f}GB total, {ram.available / (1024**3):.1f}GB available"
        except Exception as e:
            return f"Memory info unavailable: {str(e)}"
    
    async def load_model(self) -> None:
        """
        Load the NV-Reason-CXR-3B model with memory optimization.
        
        Uses 4-bit quantization for memory efficiency while maintaining
        attention weight extraction for heatmap generation.
        """
        if self.model_loaded:
            logger.info("NV-Reason-CXR model already loaded.")
            return
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading NV-Reason-CXR-3B on device: {device}")
        logger.info(f"System memory before loading NV-Reason: {self.get_system_memory_info()}")
        
        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            # 4-bit quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # Load model with 4-bit quantization
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                attn_implementation="eager",  # Required for attention scores
                device_map="auto"
            ).eval()
            
            self.model_loaded = True
            logger.info("NV-Reason-CXR-3B model loaded successfully (4-bit quantized).")
            logger.info(f"System memory after loading NV-Reason: {self.get_system_memory_info()}")
            
        except Exception as e:
            self.model_loaded = False
            logger.error(f"Failed to load NV-Reason-CXR model: {str(e)}")
            # Cleanup on failure
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    
    def _parse_structured_response(self, raw_response: str) -> NVReasonFindings:
        """
        Parse the structured <thinking> and <answer> tags from model output.
        
        NV-Reason-CXR outputs in format:
        <thinking>
        [step-by-step reasoning]
        </thinking>
        <answer>
        [final diagnosis]
        </answer>
        """
        thinking = ""
        answer = ""
        abnormalities = []
        support_devices = []
        
        try:
            # Extract thinking section
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', raw_response, re.DOTALL | re.IGNORECASE)
            if thinking_match:
                thinking = thinking_match.group(1).strip()
            
            # Extract answer section
            answer_match = re.search(r'<answer>(.*?)</answer>', raw_response, re.DOTALL | re.IGNORECASE)
            if answer_match:
                answer = answer_match.group(1).strip()
            else:
                # If no answer tags, use post-thinking content or full response
                if thinking_match:
                    remaining = raw_response[thinking_match.end():].strip()
                    answer = remaining if remaining else raw_response
                else:
                    answer = raw_response
            
            # Parse abnormalities from answer
            abnormality_keywords = [
                'pneumothorax', 'pneumonia', 'tuberculosis', 'tb', 'consolidation',
                'effusion', 'cardiomegaly', 'atelectasis', 'nodule', 'mass',
                'opacity', 'infiltrate', 'edema', 'fracture', 'lesion'
            ]
            
            answer_lower = answer.lower()
            for keyword in abnormality_keywords:
                if keyword in answer_lower:
                    # Check if it's a positive finding (not "no pneumothorax")
                    # Look for negation patterns
                    negation_pattern = rf'(no|without|absence of|normal|negative for)\s+\w*\s*{keyword}'
                    if not re.search(negation_pattern, answer_lower):
                        abnormalities.append(keyword.title())
            
            # Parse support devices
            device_keywords = [
                'endotracheal tube', 'ett', 'ng tube', 'nasogastric',
                'central line', 'picc', 'pacemaker', 'icd', 'chest tube',
                'tracheostomy', 'feeding tube', 'swan-ganz', 'catheter'
            ]
            
            for keyword in device_keywords:
                if keyword in answer_lower:
                    support_devices.append(keyword.upper() if len(keyword) <= 4 else keyword.title())
            
        except Exception as e:
            logger.error(f"Error parsing NV-Reason structured response: {e}")
            answer = raw_response
        
        return NVReasonFindings(
            thinking=thinking,
            answer=answer,
            abnormalities=list(set(abnormalities)),  # Remove duplicates
            support_devices=list(set(support_devices)),
            raw_response=raw_response
        )
    
    async def analyze_image(self, image: Image.Image) -> Tuple[str, NVReasonFindings, Image.Image]:
        """
        Analyze chest X-ray image using NV-Reason-CXR-3B.
        
        Returns:
            Tuple of (raw_response, parsed_findings, heatmap_image)
        """
        await self.load_model()
        
        try:
            # Prepare input with clinical prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Find abnormalities and support devices. Provide detailed analysis."}
                    ]
                }
            ]
            
            # Create prompt using chat template
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            
            # Process inputs
            inputs = self.processor(text=text, images=[image], return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            attentions = None
            
            # Clear cache before inference to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with torch.inference_mode():
                # First, try to get attentions from a forward pass
                try:
                    forward_outputs = self.model(
                        **inputs,
                        output_attentions=True,
                        use_cache=False
                    )
                    if hasattr(forward_outputs, 'attentions') and forward_outputs.attentions:
                        attentions = forward_outputs.attentions
                        logger.info(f"✅ Got NV-Reason attentions from forward pass: {len(attentions)} layers")
                except Exception as e:
                    logger.warning(f"NV-Reason forward pass for attentions failed: {e}")
                
                # Clear cache before generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Generate response with reduced tokens to save memory
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,  # Reduced from 2048 to save memory
                    do_sample=False,
                    output_attentions=False,  # Skip attentions in generate (we have them from forward)
                    return_dict_in_generate=True,
                    use_cache=True  # Use cache for faster generation
                )
                
                # Try to extract attentions from generation if not from forward pass
                if attentions is None:
                    for attr_name in ['attentions', 'decoder_attentions', 'cross_attentions']:
                        if hasattr(generated_ids, attr_name):
                            attr_value = getattr(generated_ids, attr_name)
                            if attr_value is not None:
                                attentions = attr_value
                                logger.info(f"✅ Got NV-Reason attentions from generation.{attr_name}")
                                break
            
            # Decode response
            input_len = inputs["input_ids"].shape[-1]
            trimmed_ids = [
                out_ids[in_len:] 
                for in_len, out_ids in zip([input_len], generated_ids.sequences)
            ]
            raw_response = self.processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            logger.info(f"NV-Reason raw response: {raw_response[:500]}...")
            
            # Parse structured response
            findings = self._parse_structured_response(raw_response)
            
            # Generate heatmap from attentions
            heatmap = self._generate_heatmap(attentions, image)
            
            return raw_response, findings, heatmap
            
        except Exception as e:
            logger.error(f"Error during NV-Reason image analysis: {str(e)}", exc_info=True)
            placeholder_heatmap = self._create_placeholder_heatmap(image)
            findings = NVReasonFindings(
                thinking="Error during analysis",
                answer=f"Analysis failed: {str(e)}",
                abnormalities=[],
                support_devices=[],
                raw_response=f"Error: {str(e)}"
            )
            return f"Error: {str(e)}", findings, placeholder_heatmap
    
    def _generate_heatmap(self, attentions: Optional[tuple], original_image: Image.Image) -> Image.Image:
        """
        Generate attention-based heatmap from Qwen2.5-VL attention weights.
        
        The Qwen2.5-VL model processes images into patches and the attention
        weights indicate which patches the model focused on.
        """
        if not attentions or not hasattr(attentions, '__len__') or len(attentions) == 0:
            logger.warning("No NV-Reason attention weights available. Creating placeholder heatmap.")
            return self._create_placeholder_heatmap(original_image)
        
        try:
            # Convert bfloat16 to float32 if needed
            if attentions[0].dtype == torch.bfloat16:
                attentions = tuple(a.float() for a in attentions)
            
            # Stack and average attention layers
            # Shape: (num_layers, batch, num_heads, seq_len, seq_len)
            stacked = torch.stack(attentions)
            
            # Average over layers and heads
            avg_attention = stacked.mean(dim=(0, 2))  # (batch, seq_len, seq_len)
            attention_map = avg_attention.squeeze(0).cpu().detach()  # (seq_len, seq_len)
            
            # For Qwen2.5-VL, try to estimate image token positions
            # The model typically uses 14x14 or similar patch grid
            seq_len = attention_map.shape[0]
            
            # Heuristic: image tokens are usually in a contiguous block
            # Try common patch sizes
            for grid_size in [14, 16, 24, 28]:
                num_image_patches = grid_size * grid_size
                if num_image_patches <= seq_len:
                    break
            else:
                grid_size = 14
                num_image_patches = 196
            
            logger.info(f"Using grid_size={grid_size} for NV-Reason heatmap (seq_len={seq_len})")
            
            # For Qwen2.5-VL, image tokens may be at the beginning or end
            # Try finding the most "interesting" block of tokens
            if seq_len >= num_image_patches:
                # Take attention from text tokens to image tokens
                # Assume image tokens are in the last num_image_patches positions
                text_len = seq_len - num_image_patches
                if text_len > 0:
                    # Average attention across text tokens to each image patch
                    text_to_image = attention_map[:text_len, -num_image_patches:]
                    image_attention = text_to_image.mean(dim=0)
                else:
                    # Use diagonal or average if mostly image tokens
                    image_attention = attention_map.mean(dim=0)[-num_image_patches:]
                
                # Reshape to grid
                if image_attention.shape[0] >= num_image_patches:
                    heatmap_np = image_attention[:num_image_patches].reshape(grid_size, grid_size).numpy()
                else:
                    # Fallback: use what we have
                    side = int(np.sqrt(image_attention.shape[0]))
                    if side * side == image_attention.shape[0]:
                        heatmap_np = image_attention.reshape(side, side).numpy()
                    else:
                        logger.warning("Cannot reshape attention to square grid, using fallback")
                        return self._create_placeholder_heatmap(original_image)
            else:
                # Sequence too short, use diagonal
                heatmap_np = attention_map.diagonal().numpy()
                side = int(np.sqrt(heatmap_np.shape[0]))
                if side * side == heatmap_np.shape[0]:
                    heatmap_np = heatmap_np.reshape(side, side)
                else:
                    return self._create_placeholder_heatmap(original_image)
            
            # Normalize heatmap
            if heatmap_np.max() > heatmap_np.min():
                heatmap_normalized = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min())
            else:
                heatmap_normalized = np.zeros_like(heatmap_np)
            
            # Create overlay
            return self._create_heatmap_overlay(heatmap_normalized, original_image)
            
        except Exception as e:
            logger.error(f"Error generating NV-Reason heatmap: {e}", exc_info=True)
            return self._create_placeholder_heatmap(original_image)
    
    def _create_heatmap_overlay(self, heatmap_np: np.ndarray, original_image: Image.Image) -> Image.Image:
        """
        Create a visually appealing heatmap overlay on the original image.
        """
        try:
            # Resize heatmap to image size
            target_size = original_image.size  # (width, height)
            heatmap_resized = cv2.resize(
                heatmap_np.astype(np.float32), 
                target_size, 
                interpolation=cv2.INTER_LINEAR
            )
            
            # Apply Gaussian blur for smoother visualization
            heatmap_blurred = cv2.GaussianBlur(
                heatmap_resized, 
                self._GAUSSIAN_BLUR_KERNEL_SIZE, 
                0
            )
            
            # Normalize to full 0-1 range for better visibility
            if heatmap_blurred.max() > heatmap_blurred.min():
                heatmap_normalized = (heatmap_blurred - heatmap_blurred.min()) / (heatmap_blurred.max() - heatmap_blurred.min())
            else:
                heatmap_normalized = np.zeros_like(heatmap_blurred)
            
            # Apply lower threshold (50th percentile) and boost contrast
            threshold = np.percentile(heatmap_normalized, 50)
            heatmap_focused = np.clip((heatmap_normalized - threshold) * 2, 0, 1)
            
            # Convert to colormap (JET colormap gives red for high attention)
            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * heatmap_focused), 
                cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Convert original image to RGB array
            original_rgb = np.array(original_image.convert("RGB"))
            
            # Blend heatmap with original using fixed alpha (not mask-weighted)
            alpha = 0.4  # Fixed alpha for consistent visibility
            blended = (1 - alpha) * original_rgb + alpha * heatmap_colored
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            
            logger.info(f"Created visible heatmap overlay with alpha={alpha}")
            return Image.fromarray(blended)
            
        except Exception as e:
            logger.error(f"Error creating heatmap overlay: {e}")
            return self._create_placeholder_heatmap(original_image)
    
    def _create_placeholder_heatmap(self, original_image: Image.Image) -> Image.Image:
        """
        Create a placeholder when attention extraction fails.
        Returns the original image with a subtle gray overlay to indicate no heatmap.
        """
        try:
            # Create a subtle overlay to indicate placeholder
            original_rgb = np.array(original_image.convert("RGB"))
            overlay = np.ones_like(original_rgb) * 128
            
            # Very subtle blend
            alpha = 0.1
            blended = (1 - alpha) * original_rgb + alpha * overlay
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            
            # Add text indicator (optional - can be removed for cleaner look)
            result = Image.fromarray(blended)
            
            logger.info("Created placeholder heatmap for NV-Reason-CXR")
            return result
            
        except Exception as e:
            logger.error(f"Error creating placeholder: {e}")
            return original_image.copy()
