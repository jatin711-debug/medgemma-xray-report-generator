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
        """Set up forward and backward hooks on the vision encoder."""
        # Find the vision encoder in the model
        # MedGemma uses SigLIP, accessed via model.vision_tower or similar
        
        vision_encoder = None
        
        # Try different possible paths to the vision encoder
        if hasattr(self.model, 'vision_tower'):
            vision_encoder = self.model.vision_tower
            logger.info("Found vision encoder at model.vision_tower")
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_tower'):
            vision_encoder = self.model.model.vision_tower
            logger.info("Found vision encoder at model.model.vision_tower")
        elif hasattr(self.model, 'vision_model'):
            vision_encoder = self.model.vision_model
            logger.info("Found vision encoder at model.vision_model")
        else:
            # Try to find it by iterating through named modules
            for name, module in self.model.named_modules():
                if 'vision' in name.lower() and hasattr(module, 'encoder'):
                    vision_encoder = module
                    logger.info(f"Found vision encoder at {name}")
                    break
        
        if vision_encoder is None:
            logger.warning("Could not find vision encoder. GradCAM will use attention fallback.")
            return
        
        # Find the last transformer block
        target_layer = None
        
        # Try to find encoder blocks
        if hasattr(vision_encoder, 'encoder'):
            encoder = vision_encoder.encoder
            if hasattr(encoder, 'layers'):
                target_layer = encoder.layers[-1]
                logger.info(f"Target layer: encoder.layers[-1] ({type(target_layer).__name__})")
            elif hasattr(encoder, 'blocks'):
                target_layer = encoder.blocks[-1]
                logger.info(f"Target layer: encoder.blocks[-1] ({type(target_layer).__name__})")
        elif hasattr(vision_encoder, 'layers'):
            target_layer = vision_encoder.layers[-1]
            logger.info(f"Target layer: vision_encoder.layers[-1]")
        elif hasattr(vision_encoder, 'blocks'):
            target_layer = vision_encoder.blocks[-1]
            logger.info(f"Target layer: vision_encoder.blocks[-1]")
        
        if target_layer is None:
            # Log available attributes for debugging
            logger.warning(f"Could not find target layer. Vision encoder attributes: {dir(vision_encoder)}")
            return
        
        self._target_layer = target_layer
        
        # Register hooks
        def forward_hook(module, input, output):
            # Handle different output formats (tuple, tensor, etc.)
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self.gradients = grad_output[0].detach()
            else:
                self.gradients = grad_output.detach()
        
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
            
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
            ]
            
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Add the image
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
                return self._create_placeholder(image)
            
            # Compute GradCAM
            cam = self._compute_gradcam()
            
            if cam is None:
                return self._create_placeholder(image)
            
            # Create overlay
            heatmap_overlay = self._create_overlay(cam, image)
            
            logger.info(f"✅ Generated GradCAM for {condition}")
            return heatmap_overlay, cam
            
        except Exception as e:
            logger.error(f"GradCAM generation failed: {e}", exc_info=True)
            return self._create_placeholder(image)
    
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
        """Compute GradCAM from stored activations and gradients."""
        try:
            # activations shape: (batch, num_patches, hidden_dim) for ViT
            # gradients shape: same as activations
            
            activations = self.activations
            gradients = self.gradients
            
            logger.info(f"Activations shape: {activations.shape}, Gradients shape: {gradients.shape}")
            
            # Global average pooling of gradients (over the hidden dimension)
            # This gives us the importance weights for each patch
            weights = gradients.mean(dim=-1, keepdim=True)  # (batch, num_patches, 1)
            
            # Weighted combination of activations
            cam = (weights * activations).sum(dim=-1)  # (batch, num_patches)
            
            # Apply ReLU (we only care about positive contributions)
            cam = F.relu(cam)
            
            # Take first batch element and reshape to grid
            cam = cam[0].cpu().numpy()  # (num_patches,)
            
            # Determine grid size (sqrt of num_patches for square grid)
            num_patches = cam.shape[0]
            
            # Handle potential CLS token (first token)
            # Most ViTs have 1 CLS token + grid_size^2 patch tokens
            sqrt_patches = int(np.sqrt(num_patches))
            if sqrt_patches * sqrt_patches == num_patches:
                grid_size = sqrt_patches
            elif (sqrt_patches) * (sqrt_patches) == num_patches - 1:
                # Has CLS token, remove it
                cam = cam[1:]
                grid_size = sqrt_patches
            else:
                # Try common grid sizes
                for gs in [14, 16, 24, 28, 32]:
                    if gs * gs == num_patches or gs * gs == num_patches - 1:
                        grid_size = gs
                        if gs * gs == num_patches - 1:
                            cam = cam[1:]
                        break
                else:
                    logger.warning(f"Cannot reshape {num_patches} patches to square grid")
                    return None
            
            cam = cam.reshape(grid_size, grid_size)
            
            # Normalize to [0, 1]
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                cam = np.zeros_like(cam)
            
            logger.info(f"GradCAM computed: {grid_size}x{grid_size}, range [{cam.min():.3f}, {cam.max():.3f}]")
            
            return cam
            
        except Exception as e:
            logger.error(f"GradCAM computation failed: {e}", exc_info=True)
            return None
    
    def _create_overlay(self, cam: np.ndarray, original_image: Image.Image) -> Image.Image:
        """Create a visualization overlay of GradCAM on the original image."""
        # Convert to numpy
        original_np = np.array(original_image.convert('RGB'))
        h, w = original_np.shape[:2]
        
        # Resize CAM to image size
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Apply colormap (JET for medical imaging)
        cam_uint8 = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend with original
        alpha = 0.4
        blended = (1 - alpha) * original_np + alpha * heatmap
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        return Image.fromarray(blended)
    
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
