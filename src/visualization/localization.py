"""
Localization and segmentation map generation for medical image analysis.

Creates localization maps with contours and segmentation overlays.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class LocalizationGenerator:
    """Generator for localization and segmentation maps on medical images."""
    
    @staticmethod
    def get_anatomical_location(x: int, y: int, img_width: int, img_height: int) -> str:
        """Determine anatomical location based on image coordinates."""
        # Normalize coordinates
        nx = x / img_width
        ny = y / img_height
        
        # Vertical zones
        if ny < 0.33:
            vertical = "upper"
        elif ny < 0.66:
            vertical = "middle"
        else:
            vertical = "lower"
        
        # Horizontal zones (PA chest X-ray: left side of image = right lung)
        if nx < 0.4:
            horizontal = "right"
        elif nx > 0.6:
            horizontal = "left"
        else:
            horizontal = "central"
        
        if horizontal == "central":
            if ny < 0.3:
                return "mediastinum"
            elif ny < 0.7:
                return "cardiac/hilar"
            else:
                return "lower central"
        else:
            return f"{horizontal} {vertical} lung"

    def generate_localization_map(
        self, 
        heatmap_raw: np.ndarray, 
        original_image: Image.Image,
        threshold_percentile: int = 60,
        min_contour_area: int = 200,
        contour_color: Tuple[int, int, int] = (0, 255, 0),
        contour_thickness: int = 2
    ) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """
        Generate lesion localization map with contours around high-attention regions.
        
        Args:
            heatmap_raw: Raw 2D attention heatmap (any size, will be resized)
            original_image: Original image to draw on
            threshold_percentile: Percentile for thresholding
            min_contour_area: Minimum contour area in pixels
            contour_color: BGR color for contours
            contour_thickness: Line thickness for contours
            
        Returns:
            Tuple of:
                - Image with contours drawn
                - List of detected regions with metadata
        """
        try:
            if heatmap_raw is None or heatmap_raw.size == 0:
                logger.warning("No heatmap provided for localization")
                return original_image.copy(), []
            
            img_width, img_height = original_image.size
            
            # Resize heatmap to image size
            heatmap_resized = cv2.resize(
                heatmap_raw.astype(np.float32), 
                (img_width, img_height), 
                interpolation=cv2.INTER_LINEAR
            )
            
            # Normalize to 0-255
            if heatmap_resized.max() > heatmap_resized.min():
                heatmap_normalized = ((heatmap_resized - heatmap_resized.min()) / 
                                     (heatmap_resized.max() - heatmap_resized.min()) * 255)
            else:
                logger.warning("Heatmap is flat, no regions to detect")
                return original_image.copy(), []
            
            heatmap_uint8 = heatmap_normalized.astype(np.uint8)
            
            # Apply Gaussian blur to smooth edges
            heatmap_smooth = cv2.GaussianBlur(heatmap_uint8, (9, 9), 0)
            
            # Threshold to binary mask
            threshold_value = np.percentile(heatmap_smooth, threshold_percentile)
            _, binary_mask = cv2.threshold(heatmap_smooth, threshold_value, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Convert original to BGR for drawing
            result_bgr = cv2.cvtColor(np.array(original_image.convert("RGB")), cv2.COLOR_RGB2BGR)
            
            # Filter and collect contours
            detected_regions = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < min_contour_area:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Get centroid
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w // 2, y + h // 2
                
                # Calculate region attention score
                mask = np.zeros(heatmap_uint8.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                region_intensity = cv2.mean(heatmap_uint8, mask=mask)[0] / 255.0
                
                # Filter: Only keep regions with >15% attention
                if region_intensity <= 0.15:
                    continue
                
                location = self.get_anatomical_location(cx, cy, img_width, img_height)
                
                detected_regions.append({
                    "region_id": i + 1,
                    "bbox": (x, y, w, h),
                    "centroid": (cx, cy),
                    "area": area,
                    "attention_score": round(region_intensity, 2),
                    "location": location,
                    "contour": contour
                })
            
            # Sort by attention score and keep only top 3
            detected_regions.sort(key=lambda r: r["attention_score"], reverse=True)
            top_regions = detected_regions[:3]
            
            # Draw the top 3 regions with labels
            for idx, region_data in enumerate(top_regions):
                contour = region_data["contour"]
                x, y, w, h = region_data["bbox"]
                attention_pct = int(region_data["attention_score"] * 100)
                
                # Draw contour
                cv2.drawContours(result_bgr, [contour], -1, contour_color, 3)
                
                # Label positioning
                label = f"R{idx+1}: {attention_pct}%"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                
                (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                
                label_x = x
                label_y = y - 10
                
                # Ensure label doesn't go off screen
                if label_y - label_h < 0:
                    label_y = y + h + label_h + 10
                if label_x + label_w > img_width:
                    label_x = img_width - label_w - 5
                
                # Draw semi-transparent background for label
                overlay = result_bgr.copy()
                cv2.rectangle(overlay, 
                            (label_x - 5, label_y - label_h - 5),
                            (label_x + label_w + 5, label_y + baseline),
                            (0, 0, 0), -1)
                result_bgr = cv2.addWeighted(overlay, 0.6, result_bgr, 0.4, 0)
                
                # Draw label text
                cv2.putText(result_bgr, label, (label_x, label_y), 
                           font, font_scale, (255, 255, 255), font_thickness)
                
                # Remove contour from region data
                region_data.pop("contour", None)
            
            # Update region IDs
            for idx, region in enumerate(top_regions):
                region["region_id"] = idx + 1
            
            result_image = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
            
            logger.info(f"Localization: showing top {len(top_regions)} regions")
            return result_image, top_regions
            
        except Exception as e:
            logger.error(f"Error generating localization map: {e}", exc_info=True)
            return original_image.copy(), []

    def generate_segmentation_map(
        self, 
        heatmap_normalized_np: Optional[np.ndarray], 
        original_image: Image.Image
    ) -> Image.Image:
        """
        Generates a segmentation map by thresholding the normalized heatmap and
        drawing contours of the thresholded regions onto the original image.

        Args:
            heatmap_normalized_np: A 2D NumPy array representing the normalized attention heatmap.
            original_image: A PIL.Image.Image object, base for drawing contours.

        Returns:
            A PIL.Image.Image object with contours drawn on it (RGB).
        """
        fallback_image = original_image.copy()

        if not isinstance(original_image, Image.Image):
            logger.error("Invalid input: original_image is not a PIL Image.")
            return fallback_image

        if heatmap_normalized_np is None:
            logger.warning("heatmap_normalized_np is None for segmentation map.")
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
                heatmap_scaled = np.zeros_like(heatmap_to_process) if min_val < 0.5 else np.ones_like(heatmap_to_process)
                logger.warning("Segmentation map input heatmap is flat.")
            else:
                heatmap_scaled = (heatmap_to_process - min_val) / (max_val - min_val)

            logger.info(f"Segmentation map input stats: min={heatmap_scaled.min():.4f}, max={heatmap_scaled.max():.4f}")

            # Resize heatmap to match the original image size
            heatmap_resized = cv2.resize(heatmap_scaled, original_image.size, interpolation=cv2.INTER_LINEAR)

            # Convert to uint8 for thresholding
            threshold_input_uint8 = np.uint8(255 * heatmap_resized)

            # Apply Otsu's thresholding
            threshold_value, binary_mask = cv2.threshold(
                threshold_input_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            logger.info(f"Otsu's threshold value: {threshold_value}")

            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            logger.info(f"Found {len(contours)} contours for segmentation map.")

            if not contours:
                logger.info("No contours found.")
                return fallback_image

            # Draw contours on the image
            original_cv_bgr = cv2.cvtColor(np.array(original_image.convert('RGB')), cv2.COLOR_RGB2BGR)
            cv2.drawContours(original_cv_bgr, contours, -1, (0, 255, 0), 2)

            segmented_image_pil = Image.fromarray(cv2.cvtColor(original_cv_bgr, cv2.COLOR_BGR2RGB))
            logger.info("Successfully generated segmentation map.")
            return segmented_image_pil

        except Exception as e:
            logger.error(f"Error generating segmentation map: {e}", exc_info=True)
            return fallback_image
