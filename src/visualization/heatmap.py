"""
Heatmap generation for medical image analysis.

Creates attention heatmap overlays to visualize AI model focus areas.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from src.core.constants import (
    MEDGEMMA_GRID_SIZE,
    MEDGEMMA_NUM_IMAGE_TOKENS,
)

logger = logging.getLogger(__name__)


class HeatmapGenerator:
    """Generator for AI attention heatmaps on medical images."""
    
    # Constants for heatmap overlay
    HEATMAP_THRESHOLD_PERCENTILE = 70  # Reduced from 95 to preserve more attention signal
    GAUSSIAN_BLUR_KERNEL_SIZE = (11, 11)  # Must be odd numbers
    HEATMAP_OVERLAY_ALPHA = 0.5  # Blending factor for the heatmap
    
    def __init__(self, processor=None):
        """
        Initialize the heatmap generator.
        
        Args:
            processor: The model processor for getting image configuration.
        """
        self.processor = processor
    
    def generate_heatmap_and_raw_normalized(
        self, 
        attentions: Optional[tuple], 
        original_image: Image.Image
    ) -> Tuple[Image.Image, Optional[np.ndarray]]:
        """
        Generates the primary AI Attention Heatmap (colored overlay) and also returns the
        raw normalized heatmap (grayscale, 0-1 range) which serves as a base for other maps.
        
        Returns:
            tuple: (heatmap_overlay_image, heatmap_raw_normalized_numpy_array)
            Returns (placeholder_overlay, None) or (placeholder_overlay, placeholder_raw_normalized) 
            on error or no attention.
        """
        if not attentions or not hasattr(attentions, '__len__') or len(attentions) == 0:
            logger.warning("No attention weights. Creating placeholder heatmap and raw map.")
            placeholder_heatmap_overlay = self.create_placeholder_heatmap(original_image)
            # Create a simple placeholder raw map
            placeholder_raw_normalized = np.ones((14, 14), dtype=np.float32) * 0.5
            return placeholder_heatmap_overlay, placeholder_raw_normalized

        try:
            if attentions[0].dtype == torch.bfloat16:
                attentions = tuple(a.float() for a in attentions)
            
            stacked_attentions = torch.stack(attentions)
            avg_attentions = stacked_attentions.mean(dim=(0, 2))
            attention_map_tensor = avg_attentions.squeeze(0).cpu().detach()

            num_image_patches = MEDGEMMA_NUM_IMAGE_TOKENS
            grid_size = MEDGEMMA_GRID_SIZE
            
            try:
                # Try to get actual values from processor if available
                if self.processor:
                    image_size_proc = self.processor.image_processor.size.get('height', 896)
                    patch_size_proc = getattr(self.processor.image_processor, 'patch_size', 14)
                    grid_size = image_size_proc // patch_size_proc
                    num_image_patches = grid_size * grid_size
                    logger.info(f"Processor config: image_size={image_size_proc}, patch_size={patch_size_proc}, grid={grid_size}x{grid_size}")
            except Exception as e_proc:
                logger.warning(f"Could not get SigLIP config ({e_proc}), using MedGemma defaults: 64x64=4096 tokens")
            
            seq_len = attention_map_tensor.shape[0]
            logger.info(f"Attention map seq_len: {seq_len}, expected image tokens: {num_image_patches}")
            
            # Image tokens are approximately at positions 5 to (5 + num_image_patches - 1)
            IMG_TOKEN_START = 5
            IMG_TOKEN_END = IMG_TOKEN_START + num_image_patches
            
            # Ensure we don't exceed sequence length
            if IMG_TOKEN_END > seq_len:
                logger.warning(f"Image token end ({IMG_TOKEN_END}) exceeds seq_len ({seq_len}). Adjusting.")
                IMG_TOKEN_END = seq_len
                actual_num_image_tokens = IMG_TOKEN_END - IMG_TOKEN_START
            else:
                actual_num_image_tokens = num_image_patches
            
            # Text tokens come after image tokens
            TEXT_TOKEN_START = IMG_TOKEN_END
            num_text_tokens = seq_len - TEXT_TOKEN_START
            
            logger.info(f"Token positions: image [{IMG_TOKEN_START}:{IMG_TOKEN_END}] ({actual_num_image_tokens} tokens), "
                       f"text [{TEXT_TOKEN_START}:{seq_len}] ({num_text_tokens} tokens)")
            
            if num_text_tokens < 1:
                logger.warning(f"Not enough text tokens ({num_text_tokens}) for heatmap. Using placeholder.")
                placeholder_heatmap_overlay = self.create_placeholder_heatmap(original_image)
                placeholder_raw_normalized = np.ones((grid_size, grid_size), dtype=np.float32) * 0.5
                return placeholder_heatmap_overlay, placeholder_raw_normalized
            
            # Extract attention FROM text tokens TO image tokens
            text_to_image_attention = attention_map_tensor[TEXT_TOKEN_START:seq_len, IMG_TOKEN_START:IMG_TOKEN_END]
            
            logger.info(f"text_to_image_attention shape: {text_to_image_attention.shape}")
            
            if text_to_image_attention.shape[1] != actual_num_image_tokens:
                logger.warning(f"Attention slice width ({text_to_image_attention.shape[1]}) != expected ({actual_num_image_tokens}). Using placeholder.")
                placeholder_heatmap_overlay = self.create_placeholder_heatmap(original_image)
                placeholder_raw_normalized = np.ones((grid_size, grid_size), dtype=np.float32) * 0.5
                return placeholder_heatmap_overlay, placeholder_raw_normalized
            
            # Average attention across all text tokens to get per-image-patch importance
            image_attention = text_to_image_attention.mean(dim=0)
            
            # Log attention statistics for debugging
            logger.info(f"Image attention stats: min={image_attention.min():.4f}, max={image_attention.max():.4f}, "
                       f"mean={image_attention.mean():.4f}, std={image_attention.std():.4f}")
            
            # Reshape to grid (sqrt check for exact fit)
            if actual_num_image_tokens != grid_size * grid_size:
                sqrt_tokens = int(actual_num_image_tokens ** 0.5)
                if sqrt_tokens * sqrt_tokens == actual_num_image_tokens:
                    grid_size = sqrt_tokens
                    logger.info(f"Adjusted grid size to {grid_size}x{grid_size}")
                else:
                    logger.warning(f"Cannot reshape {actual_num_image_tokens} tokens to square grid. Using placeholder.")
                    placeholder_heatmap_overlay = self.create_placeholder_heatmap(original_image)
                    placeholder_raw_normalized = np.ones((grid_size, grid_size), dtype=np.float32) * 0.5
                    return placeholder_heatmap_overlay, placeholder_raw_normalized
            
            heatmap_raw_np = image_attention.reshape(grid_size, grid_size).numpy()
            
            if heatmap_raw_np.max() > heatmap_raw_np.min():
                heatmap_raw_normalized = (heatmap_raw_np - heatmap_raw_np.min()) / (heatmap_raw_np.max() - heatmap_raw_np.min())
            else:
                heatmap_raw_normalized = np.zeros_like(heatmap_raw_np)
            
            logger.info(f"✅ Generated heatmap with grid {grid_size}x{grid_size}")
            heatmap_overlay_image = self.create_heatmap_overlay(np.copy(heatmap_raw_np), original_image)

            return heatmap_overlay_image, heatmap_raw_normalized

        except Exception as e:
            logger.error(f"Error generating heatmap and raw normalized map: {e}", exc_info=True)
            placeholder_heatmap_overlay = self.create_placeholder_heatmap(original_image)
            gs = 14  # Fallback grid size
            try:
                if self.processor:
                    gs = self.processor.image_processor.size['height'] // self.processor.image_processor.patch_size
            except:
                pass
            placeholder_raw_normalized = np.ones((gs, gs), dtype=np.float32) * 0.5
            return placeholder_heatmap_overlay, placeholder_raw_normalized

    def create_heatmap_overlay(
        self, 
        heatmap_raw_np: Optional[np.ndarray], 
        original_image: Image.Image, 
        is_placeholder: bool = False
    ) -> Image.Image:
        """
        Overlays a focused and visually appealing heatmap onto the original image.
        
        Args:
            heatmap_raw_np: The raw 2D NumPy array heatmap.
            original_image: The PIL.Image.Image object to overlay the heatmap on.
            is_placeholder: If True, skips the percentile thresholding.
            
        Returns:
            A PIL.Image.Image object with the heatmap overlay.
        """
        if not isinstance(original_image, Image.Image):
            logger.error("Invalid original_image provided to create_heatmap_overlay.")
            return Image.new('RGB', (224, 224), color='gray')

        if heatmap_raw_np is None or not isinstance(heatmap_raw_np, np.ndarray) or heatmap_raw_np.ndim != 2 or heatmap_raw_np.size == 0:
            logger.warning(f"Invalid heatmap_raw_np for overlay. Generating placeholder.")
            placeholder_pattern = self._generate_raw_placeholder_pattern(original_image)
            return self.create_heatmap_overlay(placeholder_pattern, original_image, is_placeholder=True)

        try:
            logger.info(f"Creating heatmap overlay. Is placeholder: {is_placeholder}")

            heatmap_processed = np.copy(heatmap_raw_np)

            if not is_placeholder:
                if heatmap_processed.max() > heatmap_processed.min():
                    threshold_value = np.percentile(heatmap_processed, self.HEATMAP_THRESHOLD_PERCENTILE)
                    heatmap_processed[heatmap_processed < threshold_value] = 0
                    logger.info(f"Applied {self.HEATMAP_THRESHOLD_PERCENTILE}th percentile threshold: {threshold_value:.4f}")

            # Normalize to 0-1 range
            min_val, max_val = heatmap_processed.min(), heatmap_processed.max()
            if max_val > min_val:
                heatmap_normalized = (heatmap_processed - min_val) / (max_val - min_val)
            else:
                heatmap_normalized = np.zeros_like(heatmap_processed)

            heatmap_uint8 = np.uint8(255 * heatmap_normalized)
            heatmap_resized = cv2.resize(heatmap_uint8, original_image.size, interpolation=cv2.INTER_LINEAR)

            # Blur slightly to smooth the heatmap
            blur_kernel = self.GAUSSIAN_BLUR_KERNEL_SIZE
            if is_placeholder and blur_kernel[0] > 5:
                blur_kernel = (max(1, blur_kernel[0]//2)*2+1, max(1, blur_kernel[1]//2)*2+1)

            heatmap_blurred = cv2.GaussianBlur(heatmap_resized, blur_kernel, 0)
            heatmap_colored = cv2.applyColorMap(heatmap_blurred, cv2.COLORMAP_JET)
            original_cv_bgr = cv2.cvtColor(np.array(original_image.convert('RGB')), cv2.COLOR_RGB2BGR)

            # Create alpha mask based on heatmap intensity
            alpha_mask = heatmap_blurred.astype(np.float32) / 255.0
            alpha_mask = np.power(alpha_mask, 0.7)  # Make low values more transparent
            alpha_mask = alpha_mask * self.HEATMAP_OVERLAY_ALPHA
            alpha_3ch = np.stack([alpha_mask, alpha_mask, alpha_mask], axis=-1)
            
            # Blend with variable alpha per pixel
            superimposed_img = (alpha_3ch * heatmap_colored + (1 - alpha_3ch) * original_cv_bgr).astype(np.uint8)
            final_image = Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
            logger.info(f"Successfully created heatmap overlay.")
            return final_image

        except Exception as e:
            logger.error(f"Error in create_heatmap_overlay: {e}", exc_info=True)
            if not is_placeholder:
                placeholder_pattern = self._generate_raw_placeholder_pattern(original_image)
                return self.create_heatmap_overlay(placeholder_pattern, original_image, is_placeholder=True)
            else:
                return Image.new('RGB', original_image.size, color='gray')

    def _generate_raw_placeholder_pattern(self, original_image: Image.Image) -> np.ndarray:
        """Generates a raw numpy array pattern for placeholder heatmaps."""
        try:
            if self.processor:
                grid_size = self.processor.image_processor.size.get('height', 224) // getattr(self.processor.image_processor, 'patch_size', 14)
            else:
                grid_size = 14
        except:
            grid_size = 14
        
        placeholder_heatmap_raw = np.zeros((grid_size, grid_size), dtype=np.float32)
        center = grid_size // 2
        for i in range(grid_size):
            for j in range(grid_size):
                dist_from_center = ((i - center)**2 + (j - center)**2)**0.5
                max_dist = (center**2 + center**2)**0.5
                placeholder_heatmap_raw[i, j] = max(0, 1 - dist_from_center / max_dist)
        return placeholder_heatmap_raw

    def create_placeholder_heatmap(self, original_image: Image.Image) -> Image.Image:
        """
        Creates a visually distinct placeholder heatmap overlay.
        
        Args:
            original_image: The original medical image.
            
        Returns:
            A placeholder heatmap overlay image.
        """
        try:
            placeholder_pattern = self._generate_raw_placeholder_pattern(original_image)
            return self.create_heatmap_overlay(placeholder_pattern, original_image, is_placeholder=True)
        except Exception as e:
            logger.error(f"Error creating placeholder heatmap: {e}", exc_info=True)
            return Image.new('RGB', original_image.size if original_image else (224, 224), color='gray')
