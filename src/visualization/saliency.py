"""
Saliency map generation for medical image analysis.

Creates saliency, density, and probability map visualizations.
"""

import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class SaliencyGenerator:
    """Generator for saliency and probability maps on medical images."""
    
    def generate_saliency_map(
        self, 
        heatmap_normalized_np: Optional[np.ndarray], 
        original_image: Image.Image
    ) -> Image.Image:
        """
        Generates a saliency map by modulating the intensity of a grayscale version
        of the original image with normalized heatmap values.

        This method highlights regions of interest (where heatmap values are high)
        while preserving the underlying structure of the original image in grayscale.

        Args:
            heatmap_normalized_np: A 2D NumPy array representing the normalized attention
                                   heatmap (ideally values between 0 and 1).
            original_image: A PIL.Image.Image object, the original image.

        Returns:
            A PIL.Image.Image object representing the saliency map (RGB).
        """
        target_size = original_image.size
        fallback_image = original_image.convert("L").convert("RGB")

        if not isinstance(original_image, Image.Image):
            logger.error("Invalid input: original_image is not a PIL Image.")
            return fallback_image

        if heatmap_normalized_np is None:
            logger.warning("heatmap_normalized_np is None for saliency map.")
            return fallback_image

        if not isinstance(heatmap_normalized_np, np.ndarray):
            logger.warning(f"Invalid input: heatmap_normalized_np is not a NumPy array.")
            return fallback_image

        if heatmap_normalized_np.ndim != 2:
            logger.warning(f"Invalid input: heatmap_normalized_np is not 2D.")
            return fallback_image

        if heatmap_normalized_np.size == 0:
            logger.warning("Invalid input: heatmap_normalized_np is empty.")
            return fallback_image

        try:
            heatmap_to_process = np.copy(heatmap_normalized_np)

            # Ensure heatmap values are scaled to 0-1
            min_val, max_val = heatmap_to_process.min(), heatmap_to_process.max()
            if min_val == max_val:
                heatmap_scaled = np.zeros_like(heatmap_to_process) if min_val == 0 else np.ones_like(heatmap_to_process) * 0.5
            else:
                heatmap_scaled = (heatmap_to_process - min_val) / (max_val - min_val)

            logger.info(f"Saliency map input stats: min={heatmap_scaled.min():.4f}, max={heatmap_scaled.max():.4f}")

            # Resize heatmap to match the original image size
            heatmap_resized = cv2.resize(heatmap_scaled, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Convert original image to grayscale
            original_gray = original_image.convert("L")
            original_gray_np = np.array(original_gray, dtype=np.float32)
            
            # Create saliency map by multiplying grayscale image with attention weights
            saliency_map_float = original_gray_np * heatmap_resized
            
            # Normalize to 0-255 range
            saliency_map_norm_float = cv2.normalize(saliency_map_float, None, 0, 255, cv2.NORM_MINMAX)
            saliency_map_uint8 = saliency_map_norm_float.astype(np.uint8)
            
            saliency_pil_image = Image.fromarray(saliency_map_uint8).convert("RGB")
            logger.info("Successfully generated saliency map.")
            return saliency_pil_image
            
        except Exception as e:
            logger.error(f"Error generating saliency map: {e}", exc_info=True)
            return fallback_image

    def generate_density_saliency_map(
        self, 
        original_image: Image.Image, 
        gaussian_kernel: int = 25,
        overlay_alpha: float = 0.4
    ) -> Image.Image:
        """
        Generates an opacity-based saliency heatmap that highlights X-ray density patterns.
        
        This is a MODEL-FREE visualization that shows:
        - Where the image is denser (bones, heart, opacities) → warmer colors (red/yellow)
        - Where the image has more air (lung fields) → cooler colors (blue)
        
        Args:
            original_image: The chest X-ray image
            gaussian_kernel: Size of Gaussian blur kernel (must be odd)
            overlay_alpha: Max alpha for the overlay (0-1)
            
        Returns:
            PIL Image with density saliency overlay
        """
        try:
            # Convert to grayscale numpy array
            gray_np = np.array(original_image.convert("L"), dtype=np.float32)
            
            # Apply Gaussian smoothing to remove noise
            if gaussian_kernel % 2 == 0:
                gaussian_kernel += 1  # Ensure odd
            smoothed = cv2.GaussianBlur(gray_np, (gaussian_kernel, gaussian_kernel), 0)
            
            # Normalize to 0-255 range
            min_val, max_val = smoothed.min(), smoothed.max()
            if max_val > min_val:
                normalized = ((smoothed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(smoothed, dtype=np.uint8)
            
            # Apply JET colormap (blue=low density, red=high density)
            heatmap_colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            
            # Convert original to BGR for blending
            original_bgr = cv2.cvtColor(np.array(original_image.convert("RGB")), cv2.COLOR_RGB2BGR)
            
            # Uniform alpha blending
            blended = cv2.addWeighted(heatmap_colored, overlay_alpha, original_bgr, 1.0 - overlay_alpha, 0)
            
            # Convert back to PIL RGB
            result = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
            
            logger.info(f"Generated density saliency map (kernel={gaussian_kernel}, alpha={overlay_alpha})")
            return result
            
        except Exception as e:
            logger.error(f"Error generating density saliency map: {e}", exc_info=True)
            return original_image.convert("RGB")

    def generate_probability_map(
        self, 
        heatmap_normalized_np: Optional[np.ndarray], 
        original_image: Image.Image
    ) -> Image.Image:
        """
        Generates a probability map visualizing normalized heatmap values as grayscale intensities.

        Args:
            heatmap_normalized_np: A 2D NumPy array representing the normalized attention heatmap.
            original_image: A PIL.Image.Image object, used for determining target size.

        Returns:
            A PIL.Image.Image object representing the probability map (RGB, grayscale content).
        """
        target_size = original_image.size
        fallback_image = original_image.convert("L").convert("RGB")

        if not isinstance(original_image, Image.Image):
            logger.error("Invalid input: original_image is not a PIL Image.")
            return fallback_image

        if heatmap_normalized_np is None:
            logger.warning("heatmap_normalized_np is None for probability map.")
            return fallback_image

        if not isinstance(heatmap_normalized_np, np.ndarray):
            logger.warning("Invalid input: heatmap_normalized_np is not a NumPy array.")
            return fallback_image

        if heatmap_normalized_np.ndim != 2:
            logger.warning("Invalid input: heatmap_normalized_np is not 2D.")
            return fallback_image

        if heatmap_normalized_np.size == 0:
            logger.warning("Invalid input: heatmap_normalized_np is empty.")
            return fallback_image

        try:
            heatmap_to_process = np.copy(heatmap_normalized_np)

            # Ensure heatmap values are scaled to 0-1
            min_val, max_val = heatmap_to_process.min(), heatmap_to_process.max()
            if min_val == max_val:
                heatmap_scaled = np.zeros_like(heatmap_to_process) if min_val == 0 else np.ones_like(heatmap_to_process) * 0.5
            else:
                heatmap_scaled = (heatmap_to_process - min_val) / (max_val - min_val)

            logger.info(f"Probability map input stats: min={heatmap_scaled.min():.4f}, max={heatmap_scaled.max():.4f}")

            # Resize heatmap to match the original image size
            heatmap_resized = cv2.resize(heatmap_scaled, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Convert to uint8 (0-255 range)
            prob_map_uint8 = np.uint8(255 * heatmap_resized)
            
            # Convert to RGB PIL Image
            prob_map_pil = Image.fromarray(prob_map_uint8).convert("RGB")
            logger.info("Successfully generated probability map.")
            return prob_map_pil
            
        except Exception as e:
            logger.error(f"Error generating probability map: {e}", exc_info=True)
            return fallback_image
