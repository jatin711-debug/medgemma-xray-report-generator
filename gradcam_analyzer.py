"""
GradCAM Implementation for MedGemma's SigLIP Vision Encoder

Provides condition-specific visual explanations for AI diagnostic decisions.
Uses gradient-weighted class activation mapping on the vision transformer.

Target Layer Strategy:
- MedGemma uses SigLIP (a ViT-based encoder) for vision
- We hook the final transformer block's output for GradCAM
- Since MedGemma is a VLM (outputs text), we use token probability as class score

Usage:
    gradcam = MedGemmaGradCAM(model, processor)
    heatmap = gradcam.generate_cam(image, condition="pneumonia")
"""

import logging
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class MedGemmaGradCAM:
    """
    GradCAM for MedGemma's SigLIP vision encoder.
    
    Since MedGemma is a VLM that outputs text, we:
    1. Hook the final vision encoder layer
    2. Use the model's text generation probability as a proxy for class score
    3. Compute gradients of this score w.r.t. the hooked activations
    4. Generate condition-specific heatmaps
    """
    
    # Keywords to detect in generated text for each condition
    CONDITION_KEYWORDS = {
        "pneumonia": ["pneumonia", "consolidation", "infiltrate", "opacity"],
        "pneumothorax": ["pneumothorax", "ptx", "collapsed lung"],
        "tuberculosis": ["tuberculosis", "tb", "cavity", "miliary"],
    }
    
    def __init__(self, model, processor):
        """
        Initialize GradCAM with MedGemma model and processor.
        
        Args:
            model: The loaded MedGemma model (AutoModelForImageTextToText)
            processor: The MedGemma processor
        """
        self.model = model
        self.processor = processor
        self.device = next(model.parameters()).device
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Find and hook the vision encoder's target layer
        self._target_layer = None
        self._hook_handles = []
        self._setup_hooks()
        
        logger.info("MedGemmaGradCAM initialized")
    
    def _setup_hooks(self):
        """Set up forward and backward hooks on the vision encoder.
        
        MedGemma-4B-IT uses MedSigLIP (SigLIP-400M variant) with:
        - 14 transformer layers
        - 896x896 input resolution
        - 16x16 grid = 256 patches
        - Hidden dim: 1024
        """
        # Try different paths to find the vision encoder
        # Based on HuggingFace transformers structure for VLMs
        
        vision_encoder = None
        encoder_path = None
        
        # List of paths to try (in order of likelihood for MedGemma)
        paths_to_try = [
            ('model.vision_tower', lambda: self.model.vision_tower),
            ('model.model.vision_tower', lambda: self.model.model.vision_tower),
            ('model.vision_tower.vision_model', lambda: self.model.vision_tower.vision_model),
            ('model.model.vision_tower.vision_model', lambda: self.model.model.vision_tower.vision_model),
            ('model.vision_model', lambda: self.model.vision_model),
            ('model.model.vision_model', lambda: self.model.model.vision_model),
        ]
        
        for path_name, path_fn in paths_to_try:
            try:
                candidate = path_fn()
                if candidate is not None:
                    vision_encoder = candidate
                    encoder_path = path_name
                    logger.info(f"Found vision encoder at {path_name}")
                    break
            except AttributeError:
                continue
        
        # Fallback: search through named modules
        if vision_encoder is None:
            for name, module in self.model.named_modules():
                # Look for SigLIP or vision-related modules
                if any(x in name.lower() for x in ['siglip', 'vision_tower', 'vision_model']):
                    if hasattr(module, 'encoder') or hasattr(module, 'layers'):
                        vision_encoder = module
                        encoder_path = name
                        logger.info(f"Found vision encoder via search at {name}")
                        break
        
        if vision_encoder is None:
            logger.warning("Could not find vision encoder. GradCAM will use attention fallback.")
            logger.info(f"Available top-level attributes: {[n for n in dir(self.model) if not n.startswith('_')]}")
            return
        
        # Now find the last transformer layer inside the encoder
        # SigLIP structure: encoder.layers[-1] (14 layers total)
        target_layer = None
        layer_path = None
        
        # Try different layer access patterns
        # For MedGemma: vision_tower.vision_model.encoder.layers[-1]
        layer_paths = [
            ('vision_model.encoder.layers', lambda e: e.vision_model.encoder.layers[-1]),
            ('vision_model.encoder.blocks', lambda e: e.vision_model.encoder.blocks[-1]),
            ('encoder.layers', lambda e: e.encoder.layers[-1]),
            ('encoder.blocks', lambda e: e.encoder.blocks[-1]),
            ('layers', lambda e: e.layers[-1]),
            ('blocks', lambda e: e.blocks[-1]),
            ('model.encoder.layers', lambda e: e.model.encoder.layers[-1] if hasattr(e, 'model') else None),
        ]
        
        for lpath_name, lpath_fn in layer_paths:
            try:
                layer = lpath_fn(vision_encoder)
                if layer is not None:
                    target_layer = layer
                    layer_path = lpath_name
                    logger.info(f"Target layer found at {encoder_path}.{lpath_name}[-1] ({type(target_layer).__name__})")
                    break
            except (AttributeError, IndexError, TypeError):
                continue
        
        if target_layer is None:
            # Log structure for debugging
            encoder_attrs = [n for n in dir(vision_encoder) if not n.startswith('_')]
            logger.warning(f"Could not find target layer. Vision encoder has: {encoder_attrs}")
            
            # Try to find any layers/blocks attribute
            if hasattr(vision_encoder, 'encoder'):
                sub_attrs = [n for n in dir(vision_encoder.encoder) if not n.startswith('_')]
                logger.warning(f"vision_encoder.encoder has: {sub_attrs}")
            return
        
        self._target_layer = target_layer
        
        # Register hooks
        def forward_hook(module, input, output):
            # Handle different output formats (tuple, tensor, BaseModelOutput, etc.)
            if hasattr(output, 'last_hidden_state'):
                self.activations = output.last_hidden_state.detach()
            elif isinstance(output, tuple):
                self.activations = output[0].detach()
            elif isinstance(output, torch.Tensor):
                self.activations = output.detach()
            else:
                logger.warning(f"Unexpected output type in forward hook: {type(output)}")
                self.activations = None
        
        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple) and len(grad_output) > 0:
                if grad_output[0] is not None:
                    self.gradients = grad_output[0].detach()
            elif isinstance(grad_output, torch.Tensor):
                self.gradients = grad_output.detach()
            else:
                logger.warning(f"Unexpected grad_output type: {type(grad_output)}")
        
        handle_fwd = target_layer.register_forward_hook(forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)
        
        self._hook_handles = [handle_fwd, handle_bwd]
        logger.info("✅ GradCAM hooks registered successfully")
    
    def _cleanup_hooks(self):
        """Remove registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
    
    def generate_cam(
        self, 
        image: Image.Image, 
        condition: str = "pneumonia",
        prompt: Optional[str] = None
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Generate GradCAM heatmap for a specific condition.
        
        Args:
            image: Input chest X-ray image (PIL)
            condition: Target condition ("pneumonia", "pneumothorax", "tuberculosis")
            prompt: Optional custom prompt (auto-generated if not provided)
            
        Returns:
            Tuple of (heatmap_overlay_image, raw_cam_numpy)
        """
        if self._target_layer is None:
            logger.warning("GradCAM not available, using placeholder")
            return self._create_placeholder(image)
        
        # Reset stored activations/gradients
        self.activations = None
        self.gradients = None
        
        # Create condition-specific prompt if not provided
        if prompt is None:
            prompt = f"Analyze this chest X-ray for {condition}. Is {condition} present?"
        
        try:
            # Prepare inputs
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Build messages with image reference
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image}, 
                    {"type": "text", "text": prompt}
                ]}
            ]
            
            # Process inputs - pass image directly to apply_chat_template
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Convert pixel_values to correct dtype
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(dtype=self.model.dtype)
            else:
                # Fallback: manually add pixel_values if not included
                pixel_values = self.processor.image_processor(image, return_tensors="pt")['pixel_values']
                inputs['pixel_values'] = pixel_values.to(self.device, dtype=self.model.dtype)
            
            # Enable gradient computation
            self.model.eval()
            
            # Forward pass with gradient tracking
            with torch.enable_grad():
                # We need to get the logits for specific tokens related to the condition
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Get logits for condition-related tokens
                logits = outputs.logits  # Shape: (batch, seq_len, vocab_size)
                
                # Create a target score based on condition keywords in vocabulary
                # We use the probability of generating words indicating the condition
                condition_score = self._compute_condition_score(logits, condition)
                
                # Backward pass
                self.model.zero_grad()
                condition_score.backward(retain_graph=True)
            
            # Check if we got activations and gradients
            if self.activations is None or self.gradients is None:
                logger.warning("Failed to capture activations/gradients")
                placeholder_img, placeholder_cam = self._create_placeholder(image)
                return placeholder_img, placeholder_cam, []
            
            # Compute GradCAM
            cam = self._compute_gradcam()
            
            if cam is None:
                placeholder_img, placeholder_cam = self._create_placeholder(image)
                return placeholder_img, placeholder_img, placeholder_cam, []
            
            # Create clinical overlay: returns (pure_heatmap, localization_map, regions)
            pure_heatmap, localization_map, detected_regions = self._create_overlay(cam, image, condition)
            
            logger.info(f"✅ Generated clinical GradCAM for {condition}: {len(detected_regions)} regions")
            # Return: (AI Evidence Map, Localization Map, raw CAM, regions)
            return pure_heatmap, localization_map, cam, detected_regions
            
        except Exception as e:
            logger.error(f"GradCAM generation failed: {e}", exc_info=True)
            placeholder_img, placeholder_cam = self._create_placeholder(image)
            return placeholder_img, placeholder_img, placeholder_cam, []
    
    def _compute_condition_score(self, logits: torch.Tensor, condition: str) -> torch.Tensor:
        """
        Compute a score representing the model's confidence in the condition.
        
        Uses the logits of the last generated token and looks for tokens
        that would indicate the presence of the condition.
        """
        # Get vocabulary
        vocab = self.processor.tokenizer.get_vocab()
        
        # Get keywords for this condition
        keywords = self.CONDITION_KEYWORDS.get(condition.lower(), [condition.lower()])
        
        # Find token IDs for keywords
        keyword_token_ids = []
        for keyword in keywords:
            # Try to find tokens that contain the keyword
            for token, token_id in vocab.items():
                if keyword.lower() in token.lower():
                    keyword_token_ids.append(token_id)
        
        # Also add common affirmative tokens (yes, detected, present, etc.)
        affirmative_words = ["yes", "detected", "present", "positive", "shows", "indicates"]
        for word in affirmative_words:
            if word in vocab:
                keyword_token_ids.append(vocab[word])
        
        keyword_token_ids = list(set(keyword_token_ids))[:50]  # Limit to prevent memory issues
        
        if not keyword_token_ids:
            # Fallback: use max logit as score
            logger.warning(f"No keyword tokens found for {condition}, using max logit")
            return logits[:, -1, :].max()
        
        # Get probabilities for condition tokens at the last position
        last_logits = logits[:, -1, :]  # (batch, vocab_size)
        probs = F.softmax(last_logits, dim=-1)
        
        # Sum probabilities of condition-related tokens
        keyword_token_ids_tensor = torch.tensor(keyword_token_ids, device=self.device)
        condition_score = probs[:, keyword_token_ids_tensor].sum()
        
        return condition_score
    
    def _compute_gradcam(self) -> Optional[np.ndarray]:
        """
        Compute clinical-grade GradCAM from stored activations and gradients.
        
        Following strict clinical requirements:
        1. Gradient-weight the patch embeddings (not raw attention)
        2. Aggregate across channels with proper weighting
        3. Reshape to spatial grid
        4. Apply spatial smoothing for regional output
        5. Threshold top 25% activations
        6. Output smooth, localized heatmap suitable for clinical reports
        """
        try:
            # activations shape: (batch, num_patches, hidden_dim) for ViT
            # gradients shape: same as activations
            
            activations = self.activations
            gradients = self.gradients
            
            logger.info(f"Activations shape: {activations.shape}, Gradients shape: {gradients.shape}")
            
            # =====================================================
            # STEP 1: Compute channel-wise importance weights
            # Global average pooling of gradients over spatial dimension
            # This gives importance weight per channel
            # =====================================================
            # Shape: (batch, hidden_dim) - importance per channel
            alpha = gradients.mean(dim=1)  # Average over patches
            
            # =====================================================
            # STEP 2: Weighted combination of activations
            # Weight each channel by its importance, then sum
            # =====================================================
            # activations: (batch, num_patches, hidden_dim)
            # alpha: (batch, hidden_dim) -> expand to (batch, 1, hidden_dim)
            weighted_activations = activations * alpha.unsqueeze(1)
            
            # Sum across channels to get per-patch importance
            cam = weighted_activations.sum(dim=-1)  # (batch, num_patches)
            
            # =====================================================
            # STEP 3: Apply ReLU (only positive contributions)
            # =====================================================
            cam = F.relu(cam)
            
            # Take first batch element and convert to numpy
            cam = cam[0].cpu().float().numpy()  # (num_patches,)
            
            # =====================================================
            # STEP 4: Reshape to spatial grid
            # =====================================================
            num_patches = cam.shape[0]
            sqrt_patches = int(np.sqrt(num_patches))
            
            # Handle CLS token if present
            if sqrt_patches * sqrt_patches == num_patches:
                grid_size = sqrt_patches
            elif sqrt_patches * sqrt_patches == num_patches - 1:
                cam = cam[1:]  # Remove CLS token
                grid_size = sqrt_patches
            else:
                # Try common grid sizes
                for gs in [14, 16, 24, 28, 32, 64]:
                    if gs * gs == num_patches:
                        grid_size = gs
                        break
                    elif gs * gs == num_patches - 1:
                        cam = cam[1:]
                        grid_size = gs
                        break
                else:
                    logger.warning(f"Cannot reshape {num_patches} patches to square grid")
                    return None
            
            cam = cam.reshape(grid_size, grid_size)
            
            # =====================================================
            # STEP 5: Spatial smoothing for regional output
            # Use Gaussian blur to create smooth, non-speckled heatmap
            # =====================================================
            # Normalize first
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                cam = np.zeros_like(cam)
            
            # Apply Gaussian smoothing (kernel size relative to grid)
            kernel_size = max(3, grid_size // 8)
            if kernel_size % 2 == 0:
                kernel_size += 1  # Must be odd
            cam_smoothed = cv2.GaussianBlur(cam.astype(np.float32), (kernel_size, kernel_size), 0)
            
            # =====================================================
            # STEP 6: Threshold top 25% activations
            # Zero out low activations for clean clinical output
            # =====================================================
            threshold = np.percentile(cam_smoothed, 75)  # Keep top 25%
            cam_thresholded = np.where(cam_smoothed >= threshold, cam_smoothed, 0)
            
            # Re-normalize after thresholding
            if cam_thresholded.max() > 0:
                cam_final = cam_thresholded / cam_thresholded.max()
            else:
                cam_final = cam_smoothed  # Fallback to smoothed if all zeros
            
            logger.info(f"GradCAM computed: {grid_size}x{grid_size}, "
                       f"smoothed (kernel={kernel_size}), threshold@75%, "
                       f"range [{cam_final.min():.3f}, {cam_final.max():.3f}]")
            
            return cam_final
            
        except Exception as e:
            logger.error(f"GradCAM computation failed: {e}", exc_info=True)
            return None
    
    def _create_lung_mask(self, image_np: np.ndarray) -> np.ndarray:
        """
        Create approximate lung region mask using intensity-based segmentation.
        
        This filters out tubes, wires, and non-lung structures by:
        1. Thresholding to find dark lung regions
        2. Morphological operations to clean up
        3. Finding the two largest contours (left/right lungs)
        """
        h, w = image_np.shape[:2]
        
        # Convert to grayscale if needed
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np.copy()
        
        # Normalize to 0-255
        if gray.max() > 1:
            gray_norm = gray.astype(np.uint8)
        else:
            gray_norm = (gray * 255).astype(np.uint8)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_norm)
        
        # Threshold: lungs are typically dark regions (lower intensity)
        # Use Otsu's method to find optimal threshold
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours and keep the two largest (left and right lungs)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 1:
            # Fallback: create central elliptical mask
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (w//2, int(h*0.45)), (int(w*0.4), int(h*0.35)), 0, 0, 360, 255, -1)
            return mask
        
        # Sort by area and keep top 2
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        
        # Create mask from largest contours
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, 255, -1)
        
        # Dilate slightly to include lung boundaries
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)
        
        # Smooth the mask edges
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return mask
    
    def _create_overlay(
        self, 
        cam: np.ndarray, 
        original_image: Image.Image,
        condition: str = "Finding"
    ) -> Tuple[Image.Image, List[Dict]]:
        """
        Create clinical-grade visualization overlay with contours.
        
        Pipeline:
        1. Resize CAM to image size
        2. Apply lung mask (filter out tubes/wires)
        3. Threshold top 20% activations
        4. Smooth for regional output
        5. Extract contours for localization
        6. Blend with low opacity (0.35)
        
        Returns:
            Tuple of (overlay_image, detected_regions)
        """
        original_np = np.array(original_image.convert('RGB'))
        h, w = original_np.shape[:2]
        
        # =====================================================
        # STEP 1: Resize CAM to image size
        # =====================================================
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # =====================================================
        # STEP 2: Apply lung mask (filter artifacts)
        # =====================================================
        lung_mask = self._create_lung_mask(original_np)
        lung_mask_float = lung_mask.astype(np.float32) / 255.0
        cam_masked = cam_resized * lung_mask_float
        
        # =====================================================
        # STEP 3: Threshold top 20% activations
        # =====================================================
        if cam_masked.max() > 0:
            cam_normalized = cam_masked / cam_masked.max()
        else:
            cam_normalized = cam_masked
            
        threshold = np.percentile(cam_normalized[cam_normalized > 0], 80) if cam_normalized.max() > 0 else 0.5
        cam_thresholded = np.where(cam_normalized >= threshold, cam_normalized, 0)
        
        # =====================================================
        # STEP 4: Smooth for regional output
        # =====================================================
        smooth_kernel = max(25, min(w, h) // 20)
        if smooth_kernel % 2 == 0:
            smooth_kernel += 1
        cam_smooth = cv2.GaussianBlur(cam_thresholded.astype(np.float32), (smooth_kernel, smooth_kernel), 0)
        
        # Re-normalize
        if cam_smooth.max() > cam_smooth.min():
            cam_smooth = (cam_smooth - cam_smooth.min()) / (cam_smooth.max() - cam_smooth.min())
        
        # =====================================================
        # STEP 5: Extract contours for localization
        # =====================================================
        cam_uint8 = (cam_smooth * 255).astype(np.uint8)
        contour_thresh = int(np.percentile(cam_uint8[cam_uint8 > 0], 50)) if cam_uint8.max() > 0 else 128
        _, binary = cv2.threshold(cam_uint8, contour_thresh, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and rank contours
        detected_regions = []
        min_area = (w * h) * 0.005  # Minimum 0.5% of image
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Get bounding box and centroid
            x, y, bw, bh = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            else:
                cx, cy = x + bw//2, y + bh//2
            
            # Calculate region intensity
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            region_intensity = cv2.mean(cam_smooth, mask=mask)[0]
            
            detected_regions.append({
                "contour": contour,
                "area": area,
                "bbox": (x, y, bw, bh),
                "centroid": (cx, cy),
                "intensity": region_intensity,
                "location": self._get_anatomical_location(cx, cy, w, h)
            })
        
        # Sort by intensity and keep top 3
        detected_regions.sort(key=lambda r: r["intensity"], reverse=True)
        top_regions = detected_regions[:3]
        
        # =====================================================
        # STEP 6: Create overlay with reduced opacity (0.35)
        # =====================================================
        # Apply colormap
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Variable alpha: only activated areas get color (max 0.35 opacity)
        alpha_mask = cam_smooth ** 0.7  # Gentler falloff
        alpha_mask = np.clip(alpha_mask * 0.35, 0, 0.35)  # Reduced max opacity
        alpha_3ch = np.stack([alpha_mask, alpha_mask, alpha_mask], axis=-1)
        
        # Blend to create PURE HEATMAP (no contours) for "AI Evidence Map"
        pure_heatmap = (alpha_3ch * heatmap + (1 - alpha_3ch) * original_np).astype(np.uint8)
        
        # =====================================================
        # STEP 7: Create LOCALIZATION MAP with contours
        # =====================================================
        # Copy the blended image for localization (with contours)
        localization_map = pure_heatmap.copy()
        
        for idx, region in enumerate(top_regions):
            contour = region["contour"]
            # Green contour for detected regions
            cv2.drawContours(localization_map, [contour], -1, (0, 255, 0), 2)
            
            # Add region label (R1, R2, etc.)
            cx, cy = region["centroid"]
            label = f"R{idx + 1}"
            
            # Draw label with background
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(localization_map, (cx - 2, cy - lh - 4), (cx + lw + 4, cy + 4), (0, 0, 0), -1)
            cv2.putText(localization_map, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        logger.info(f"Clinical CAM: lung-masked, top-20%, {len(top_regions)} regions")
        
        # Return BOTH images: pure heatmap for AI Evidence, localization with contours
        return Image.fromarray(pure_heatmap), Image.fromarray(localization_map), top_regions
    
    def _get_anatomical_location(self, x: int, y: int, w: int, h: int) -> str:
        """Determine anatomical location from coordinates."""
        # Horizontal: left/central/right
        if x < w * 0.35:
            horiz = "Right"  # Radiographic convention: right side of patient
        elif x > w * 0.65:
            horiz = "Left"
        else:
            horiz = "Central"
        
        # Vertical: upper/mid/lower
        if y < h * 0.35:
            vert = "Upper"
        elif y > h * 0.65:
            vert = "Lower"
        else:
            vert = "Mid"
        
        return f"{vert} {horiz} Lung Zone"
    
    def _create_placeholder(self, image: Image.Image) -> Tuple[Image.Image, np.ndarray]:
        """Create placeholder when GradCAM fails."""
        original_np = np.array(image.convert('RGB'))
        h, w = original_np.shape[:2]
        
        # Light gray overlay
        overlay = np.ones_like(original_np) * 200
        alpha = 0.1
        blended = (1 - alpha) * original_np + alpha * overlay
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        # Return uniform CAM
        cam = np.ones((16, 16), dtype=np.float32) * 0.5
        
        return Image.fromarray(blended), cam
    
    def generate_all_conditions(
        self, 
        image: Image.Image
    ) -> Dict[str, Tuple[Image.Image, np.ndarray]]:
        """
        Generate GradCAM heatmaps for all tracked conditions.
        
        Args:
            image: Input chest X-ray image
            
        Returns:
            Dict mapping condition name to (heatmap_image, raw_cam)
        """
        results = {}
        
        for condition in self.CONDITION_KEYWORDS.keys():
            logger.info(f"Generating GradCAM for {condition}...")
            heatmap, cam = self.generate_cam(image, condition)
            results[condition] = (heatmap, cam)
        
        return results
    
    def __del__(self):
        """Cleanup hooks on destruction."""
        self._cleanup_hooks()
