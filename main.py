from builtins import str
import os
import cv2
import numpy as np
import uuid
import re
import io
import gc
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import logging
os.environ['TORCH_COMPILE_DISABLE'] = '1'
import aiohttp
import validators
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")  # Load environment variables from .env file

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, HttpUrl
from PIL import Image
from report_generator import UltimatePDFGenerator
import gc
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global analyzer instance
analyzer = None

from db import initialize_database, store_analysis_result
from aws_utils import upload_to_s3 # Import the new S3 upload function

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Medical Image Analysis API (Med-GEMMA-3 only)...")
    global analyzer
    analyzer = MedicalImageAnalyzer()
    await analyzer.load_model()
    yield
    # Shutdown - cleanup if needed
    logger.info("Shutting down Medical Image Analysis API...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Medical Image Analysis API (Med-GEMMA-3 only)",
    description="AI-powered medical image analysis using Med-GEMMA-3",
    version="1.0.0",
    lifespan=lifespan
)

pdf_generator = UltimatePDFGenerator("reports")


# Request models
class ImageURLRequest(BaseModel):
    """Request model for analyzing medical images from URLs.
    
    Attributes:
        image_url: HTTP/HTTPS URL of the medical image to analyze
    """
    image_url: HttpUrl = Field(description="HTTP/HTTPS URL of the image to analyze")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
            }
        }
    }


class ImageURLReportRequest(BaseModel):
    """Request model for analyzing medical images from URLs and generating PDF reports.
    
    Attributes:
        image_url: HTTP/HTTPS URL of the medical image to analyze and generate report for
    """
    image_url: HttpUrl = Field(description="HTTP/HTTPS URL of the image to analyze and generate report")
    image_uuid: str = Field(
        description="Unique identifier for the X-ray image",
        example="123e4567-e89b-12d3-a456-426614174000"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png",
                "image_uuid": "123e4567-e89b-12d3-a456-426614174000"
            }
        }
    }


# Response models
class RadiologyFindings(BaseModel):
    anatomical_structures: list[str] = Field(description="Identified anatomical structures")
    abnormalities: list[str] = Field(description="Detected abnormalities or pathologies")
    impression: str = Field(description="Overall clinical impression")
    recommendations: list[str] = Field(description="Clinical recommendations")
    confidence_level: str = Field(description="AI confidence level (High/Medium/Low)")


class AnalysisResponse(BaseModel):
    analysis_id: str = Field(description="Unique analysis identifier")
    timestamp: str = Field(description="Analysis timestamp")
    findings: RadiologyFindings
    raw_response: str = Field(description="Raw AI model response")
    image_metadata: Dict[str, Any] = Field(description="Image metadata")


class PDFGenerationResponse(BaseModel):
    pdf_path: str = Field(description="Path to generated PDF report")
    analysis_id: str = Field(description="Analysis identifier")


class ModelStatus(BaseModel):
    loaded: bool = Field(description="Whether the model is loaded")
    device: str = Field(description="Device the model is running on")
    memory_usage: Optional[str] = Field(description="Memory usage information")
    error: Optional[str] = Field(description="Any loading errors")


# Global state for model‐loading error
model_loading_error: Optional[str] = None


class MedicalImageAnalyzer:
    """
    Medical image analysis system using Med-GEMMA-3 for AI-powered radiology insights.
    
    This class handles the complete pipeline for medical image analysis including:
    - Model loading and management with CPU/GPU optimization
    - Image preprocessing and validation
    - AI-powered analysis using Med-GEMMA-3
    - PDF report generation
    - Memory management for optimal performance
    
    Attributes:
        model_id: The Hugging Face model identifier for Med-GEMMA-3
        pipeline: The loaded transformers pipeline for image-text-to-text
        reports_dir: Directory for storing generated PDF reports
        max_image_size: Maximum allowed image size in bytes (10MB)
        allowed_extensions: Set of supported image file extensions
        model_loaded: Boolean flag indicating if the model is successfully loaded
    """
    
    def __init__(self):
        # Only one model: Med-GEMMA-3
        self.model_id = "google/medgemma-4b-it"
        self.processor: Optional[Any] = None
        self.model: Optional[Any] = None
        self.reports_dir = "reports"
        self.max_image_size = 10 * 1024 * 1024  # 10MB
        self.allowed_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp'}
        self.model_loaded = False
        self.ensure_reports_directory()
        
    def ensure_reports_directory(self) -> None:
        """
        Ensure the reports directory exists for PDF storage.
        
        Creates the reports directory if it doesn't exist to prevent
        file operation errors during PDF generation.
        """
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)
    
    def get_system_memory_info(self) -> str:
        """
        Retrieve system memory information for monitoring and optimization.
        
        Returns detailed memory usage for either GPU (if CUDA available) or CPU.
        This information is crucial for monitoring resource usage and preventing
        out-of-memory errors during model inference.
        
        Returns:
            str: Formatted string containing memory information
            
        Note:
            Requires psutil package for CPU memory monitoring on non-CUDA systems.
        """
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
        Load the Med-GEMMA-3 model and processor with memory optimization.
        
        This method handles the loading of the Med-GEMMA-3 model and its
        associated processor. It includes several optimizations for memory
        management, including:
        - 4-bit quantization to reduce model size
        - Support for CPU offloading if CUDA is not available
        - Automatic garbage collection to free up unused memory
        
        The method will attempt to load the model on a CUDA-enabled GPU if
        available, falling back to the CPU if necessary.
        """
        global model_loading_error
        if self.model_loaded:
            logger.info("Model already loaded.")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model on device: {device}")
        logger.info(f"System memory before loading: {self.get_system_memory_info()}")

        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",  # Required for attention scores
                device_map="auto"  # Automatically handle device placement
            )
            self.model_loaded = True
            model_loading_error = None
            logger.info("Model loaded successfully.")
            logger.info(f"System memory after loading: {self.get_system_memory_info()}")
        except Exception as e:
            self.model_loaded = False
            model_loading_error = f"Failed to load model: {str(e)}"
            logger.error(model_loading_error)
            # Clean up memory on failure
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def parse_ai_response(self, raw_response: str) -> RadiologyFindings:
        """
        Parse raw AI model output for pneumothorax detection.
        
        This method intelligently extracts the finding and confidence level for pneumothorax.
        
        Args:
            raw_response: Raw text output from the Med-GEMMA-3 model
              Returns:
            RadiologyFindings: Structured medical findings for pneumothorax.
        """
        try:
            # Use regex to find confidence percentage
            confidence_match = re.search(r'(\d+\.?\d*)\s?%', raw_response)
            confidence = f"{confidence_match.group(1)}%" if confidence_match else "N/A"

            # Determine finding based on model output
            if "no pneumothorax detected" in raw_response.lower():
                finding = "No Pneumothorax Detected"
                abnormalities = ["No evidence of pneumothorax found."]
            else:
                finding = "Pneumothorax Detected"
                abnormalities = ["Evidence of pneumothorax found."]

            return RadiologyFindings(
                anatomical_structures=["Chest structures analyzed for pneumothorax."],
                abnormalities=abnormalities,
                impression=finding,
                recommendations=["Clinical correlation and expert review is recommended."],
                confidence_level=confidence
            )
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {str(e)}")
            return RadiologyFindings(
                anatomical_structures=["Analysis completed"],
                abnormalities=["Unable to parse detailed findings"],
                impression=raw_response,
                recommendations=["Manual review recommended"],
                confidence_level="Not available"
            )
    
    async def analyze_image(self, image: Image.Image) -> tuple[str, RadiologyFindings, Image.Image, Image.Image, Image.Image, Image.Image]:
        """
        Perform AI-powered medical image analysis using Med-GEMMA.
        Returns raw text response, structured findings, AI Attention Heatmap,
        New Saliency Map, Probability Map, and Segmentation Map.
        """
        await self.load_model()

        try:
            # Resize image to match model input expectations (usually 224x224)
            image_for_model = image.resize((224, 224))

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a highly experienced radiologist. You analyze chest X-rays for critical findings, especially pneumothorax."}]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Review this chest X-ray carefully for signs of pneumothorax, including loss of lung markings, "
                                "pleural line visibility, and increased radiolucency.\n\n"
                                "Respond strictly in the following format:\n\n"
                                "**Finding**: [Pneumothorax detected | No pneumothorax detected]\n"
                                "**Confidence**: [XX%]\n"
                                "**Brief Reasoning**: [One-sentence justification using clinical language]\n\n"
                                "Keep the response within 2 lines total. Do not include any disclaimers or extra comments."
                            )
                        },
                        {"type": "image", "image": image_for_model}
                    ]
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

            input_len = inputs["input_ids"].shape[-1]
            attentions = None

            with torch.inference_mode():
                try:
                    forward_outputs = self.model(
                        **inputs,
                        output_attentions=True,
                        use_cache=False
                    )
                    if hasattr(forward_outputs, 'attentions') and forward_outputs.attentions:
                        attentions = forward_outputs.attentions
                        logger.info(f"✅ Got attentions from forward pass: {len(attentions)} layers")
                    else:
                        logger.warning("No attentions from forward pass")
                except Exception as e:
                    logger.warning(f"Forward pass failed: {e}")

                generation_output = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    output_attentions=True,
                    return_dict_in_generate=True,
                    use_cache=False
                )

                if attentions is None:
                    attention_attrs = ['attentions', 'decoder_attentions', 'cross_attentions', 'encoder_attentions']
                    for attr_name in attention_attrs:
                        if hasattr(generation_output, attr_name):
                            attr_value = getattr(generation_output, attr_name)
                            if attr_value is not None:
                                attentions = attr_value
                                logger.info(f"✅ Got attentions from generation.{attr_name}: {len(attentions) if hasattr(attentions, '__len__') else 'unknown length'}")
                                break
                    if attentions is None:
                        logger.warning("No attentions found in generation output")
                        available_attrs = [attr for attr in dir(generation_output) if not attr.startswith('_')]
                        logger.info(f"Available generation output attributes: {available_attrs}")

            generation = generation_output.sequences[0][input_len:]
            raw_response = self.processor.decode(generation, skip_special_tokens=True)

            if attentions is not None:
                logger.info(f"Attention info - Type: {type(attentions)}, Length: {len(attentions) if hasattr(attentions, '__len__') else 'No len'}")
                if hasattr(attentions, '__len__') and len(attentions) > 0:
                    first_attention = attentions[0] if isinstance(attentions, (list, tuple)) else attentions
                    if hasattr(first_attention, 'shape'):
                        logger.info(f"First attention shape: {first_attention.shape}")
            else:
                logger.warning("⚠️  Still no attentions found - will create placeholder maps")

            # Generate various maps
            # heatmap_raw_normalized is the key intermediate map (normalized attention)
            heatmap_image, heatmap_raw_normalized = self._generate_heatmap_and_raw_normalized(attentions, image)

            if heatmap_raw_normalized is None: # Handle case where attention extraction failed
                logger.warning("heatmap_raw_normalized is None. Generating placeholder for all maps.")
                # Create a default small gray square as placeholder for heatmap_raw_normalized
                # The size (e.g., 14x14) should ideally match what would have been the grid_size.
                # Using a common fallback like 14x14.
                heatmap_raw_normalized = np.ones((14, 14), dtype=np.float32) * 0.5
                if heatmap_image is None: # If _generate_heatmap_and_raw_normalized also returned None for image
                     heatmap_image = self._create_placeholder_heatmap(image)


            new_saliency_map_image = self._generate_new_saliency_map(heatmap_raw_normalized, image)
            probability_map_image = self._generate_probability_map(heatmap_raw_normalized, image)
            segmentation_map_image = self._generate_segmentation_map(heatmap_raw_normalized, image)

            findings = self.parse_ai_response(raw_response)
            return raw_response, findings, heatmap_image, new_saliency_map_image, probability_map_image, segmentation_map_image

        except Exception as e:
            logger.error(f"Error during image analysis: {str(e)}", exc_info=True)
            # Return placeholders if analysis fails critically before map generation
            placeholder_img = self._create_placeholder_heatmap(image)
            findings = RadiologyFindings(
                anatomical_structures=["Error during analysis"],
                abnormalities=[f"Error: {str(e)}"],
                impression="Analysis Incomplete",
                recommendations=["Review error logs"],
                confidence_level="N/A"
            )
            # Ensure 6 items are returned by analyze_image in all paths
            return f"Error: {str(e)}", findings, placeholder_img, placeholder_img, placeholder_img, placeholder_img


    def _generate_heatmap_and_raw_normalized(self, attentions: Optional[tuple], original_image: Image.Image) -> tuple[Image.Image, Optional[np.ndarray]]:
        """
        Generates the primary AI Attention Heatmap (colored overlay) and also returns the
        raw normalized heatmap (grayscale, 0-1 range) which serves as a base for other maps.
        Returns (heatmap_overlay_image, heatmap_raw_normalized_numpy_array)
        Returns (placeholder_overlay, None) or (placeholder_overlay, placeholder_raw_normalized) on error or no attention.
        """
        if not attentions or not hasattr(attentions, '__len__') or len(attentions) == 0:
            logger.warning("No attention weights. Creating placeholder heatmap and raw map.")
            placeholder_heatmap_overlay = self._create_placeholder_heatmap(original_image)
            # Create a simple placeholder raw map (e.g., a small gray square of a default grid size)
            placeholder_raw_normalized = np.ones((14,14), dtype=np.float32) * 0.5
            return placeholder_heatmap_overlay, placeholder_raw_normalized

        try:
            if attentions[0].dtype == torch.bfloat16:
                attentions = tuple(a.float() for a in attentions)
            
            stacked_attentions = torch.stack(attentions)
            avg_attentions = stacked_attentions.mean(dim=(0, 2))
            attention_map_tensor = avg_attentions.squeeze(0).cpu().detach()

            try:
                image_size_proc = self.processor.image_processor.size['height']
                patch_size_proc = self.processor.image_processor.patch_size
                grid_size = image_size_proc // patch_size_proc
                num_image_patches = grid_size * grid_size
            except Exception as e_proc:
                logger.warning(f"Could not get SigLIP config from processor ({e_proc}), using defaults.")
                grid_size = 14
                num_image_patches = grid_size * grid_size
            
            seq_len = attention_map_tensor.shape[0]
            # Ensure num_text_tokens is calculated correctly based on actual sequence length from model
            # The image patches might not always be 'num_image_patches' if seq_len is smaller
            # This happens if the text part of the input is very short.
            # Let's assume the image part is `num_image_patches` and text is `seq_len - num_image_patches`
            # However, the actual number of image tokens in attention_map_tensor might be related to `grid_size*grid_size`
            # The critical part is `text_to_image_attention = attention_map_tensor[:-num_image_patches, -num_image_patches:]`
            # Let's assume the last `num_image_patches` tokens are image tokens.
            
            actual_num_image_tokens_in_map = num_image_patches
            if seq_len < num_image_patches:
                 logger.warning(f"Sequence length ({seq_len}) is smaller than expected image patches ({num_image_patches}). Adjusting.")
                 actual_num_image_tokens_in_map = seq_len # Cannot have more image tokens than seq_len

            num_text_tokens = seq_len - actual_num_image_tokens_in_map
            
            if num_text_tokens < 1 : # No text tokens to average attention from
                logger.warning(f"Not enough text tokens ({num_text_tokens}) for heatmap based on seq_len {seq_len} and image_tokens {actual_num_image_tokens_in_map}. Using placeholder.")
                placeholder_heatmap_overlay = self._create_placeholder_heatmap(original_image)
                placeholder_raw_normalized = np.ones((grid_size, grid_size), dtype=np.float32) * 0.5
                return placeholder_heatmap_overlay, placeholder_raw_normalized

            # Attention from text tokens to image tokens
            # Text tokens are assumed to be first, image tokens last.
            text_to_image_attention = attention_map_tensor[:num_text_tokens, -actual_num_image_tokens_in_map:]

            if text_to_image_attention.shape[1] != num_image_patches: # Check if the slice width matches expected patches
                 logger.warning(f"Sliced text_to_image_attention width ({text_to_image_attention.shape[1]}) "
                                f"does not match num_image_patches ({num_image_patches}). Using placeholder.")
                 placeholder_heatmap_overlay = self._create_placeholder_heatmap(original_image)
                 placeholder_raw_normalized = np.ones((grid_size, grid_size), dtype=np.float32) * 0.5
                 return placeholder_heatmap_overlay, placeholder_raw_normalized

            image_attention = text_to_image_attention.mean(dim=0) # Average attention from all text tokens to each image patch

            if image_attention.shape[0] != num_image_patches: # Final check on the vector length
                 logger.warning(f"Resulting image attention vector length ({image_attention.shape[0]}) "
                                f"does not match num_image_patches ({num_image_patches}). Using placeholder.")
                 placeholder_heatmap_overlay = self._create_placeholder_heatmap(original_image)
                 placeholder_raw_normalized = np.ones((grid_size, grid_size), dtype=np.float32) * 0.5
                 return placeholder_heatmap_overlay, placeholder_raw_normalized

            heatmap_raw_np = image_attention.reshape(grid_size, grid_size).numpy()

            if heatmap_raw_np.max() > heatmap_raw_np.min():
                heatmap_raw_normalized = (heatmap_raw_np - heatmap_raw_np.min()) / (heatmap_raw_np.max() - heatmap_raw_np.min())
            else:
                heatmap_raw_normalized = np.zeros_like(heatmap_raw_np)

            heatmap_overlay_image = self._create_heatmap_overlay(np.copy(heatmap_raw_np), original_image)

            return heatmap_overlay_image, heatmap_raw_normalized

        except Exception as e:
            logger.error(f"Error generating heatmap and raw normalized map: {e}", exc_info=True)
            placeholder_heatmap_overlay = self._create_placeholder_heatmap(original_image)
            # Attempt to create a placeholder raw map with a default grid size if possible
            try:
                gs = self.processor.image_processor.size['height'] // self.processor.image_processor.patch_size
            except:
                gs = 14 # Fallback grid size
            placeholder_raw_normalized = np.ones((gs,gs), dtype=np.float32) * 0.5
            return placeholder_heatmap_overlay, placeholder_raw_normalized

    def _generate_new_saliency_map(self, heatmap_normalized_np: Optional[np.ndarray], original_image: Image.Image) -> Image.Image:
        """Generates a saliency map by overlaying a heatmap on a grayscale version of the original image."""
        target_size = original_image.size
        if heatmap_normalized_np is None:
            logger.warning("heatmap_normalized_np is None for new saliency map. Returning grayscale original.")
            return original_image.convert("L").convert("RGB")

        try:
            if heatmap_normalized_np.ndim != 2:
                logger.error(f"heatmap_normalized_np is not 2D for saliency map, shape: {heatmap_normalized_np.shape}. Returning placeholder.")
                return original_image.convert("L").convert("RGB")

            # Log statistics for debugging
            logger.info(f"Saliency map input stats: min={heatmap_normalized_np.min():.4f}, max={heatmap_normalized_np.max():.4f}, mean={heatmap_normalized_np.mean():.4f}")

            # Resize heatmap to match the original image size
            heatmap_resized = cv2.resize(heatmap_normalized_np, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Convert original image to grayscale
            original_gray = original_image.convert("L")
            original_gray_np = np.array(original_gray)
            
            # Create saliency map by multiplying grayscale image with attention weights
            # This highlights important regions while preserving image structure
            saliency_map = original_gray_np.astype(np.float32) * heatmap_resized
            
            # Normalize to 0-255 range
            saliency_map = np.clip(saliency_map, 0, 255).astype(np.uint8)
            
            # Convert back to RGB PIL Image
            return Image.fromarray(saliency_map).convert("RGB")
            
        except Exception as e:
            logger.error(f"Error generating new saliency map: {e}", exc_info=True)
            return original_image.convert("L").convert("RGB")

    def _generate_probability_map(self, heatmap_normalized_np: Optional[np.ndarray], original_image: Image.Image) -> Image.Image:
        """Generates a probability map showing attention weights as intensity values."""
        target_size = original_image.size
        if heatmap_normalized_np is None:
            logger.warning("heatmap_normalized_np is None for probability map. Returning grayscale original.")
            return original_image.convert("L").convert("RGB")
        try:
            if heatmap_normalized_np.ndim != 2:
                logger.error(f"heatmap_normalized_np is not 2D for probability map, shape: {heatmap_normalized_np.shape}. Returning placeholder.")
                return original_image.convert("L").convert("RGB")

            # Log statistics for debugging
            logger.info(f"Probability map input stats: min={heatmap_normalized_np.min():.4f}, max={heatmap_normalized_np.max():.4f}, mean={heatmap_normalized_np.mean():.4f}")

            # Resize heatmap to match the original image size
            heatmap_resized = cv2.resize(heatmap_normalized_np, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Convert to uint8 (0-255 range) - this represents probability/attention strength
            prob_map = np.uint8(255 * heatmap_resized)
            
            # Convert to RGB PIL Image (grayscale probability map)
            return Image.fromarray(prob_map).convert("RGB")
            
        except Exception as e:
            logger.error(f"Error generating probability map: {e}", exc_info=True)
            return original_image.convert("L").convert("RGB")

    def _generate_segmentation_map(self, heatmap_normalized_np: Optional[np.ndarray], original_image: Image.Image) -> Image.Image:
        """Generates a segmentation map by thresholding the normalized heatmap and drawing contours."""
        if heatmap_normalized_np is None:
            logger.warning("heatmap_normalized_np is None for segmentation map. Returning original image as placeholder.")
            return original_image.copy()

        try:
            # Ensure heatmap_normalized_np is 2D before resizing
            if heatmap_normalized_np.ndim != 2:
                logger.error(f"heatmap_normalized_np is not 2D for segmentation map, shape: {heatmap_normalized_np.shape}. Returning original image.")
                return original_image.copy()

            heatmap_resized_for_segmentation = cv2.resize(heatmap_normalized_np, original_image.size, interpolation=cv2.INTER_LINEAR)

            # Ensure it's in 0-1 range before multiplying by 255
            if heatmap_resized_for_segmentation.max() > 1.0 or heatmap_resized_for_segmentation.min() < 0.0:
                 heatmap_resized_for_segmentation = np.clip(heatmap_resized_for_segmentation, 0.0, 1.0)

            threshold_input = np.uint8(255 * heatmap_resized_for_segmentation)

            # Otsu's thresholding might be more adaptive here
            threshold_value_seg, binary_mask = cv2.threshold(threshold_input, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # As an alternative, fixed percentile:
            # threshold_value_fixed = np.percentile(heatmap_resized_for_segmentation, 85)
            # _, binary_mask = cv2.threshold(threshold_input, int(255 * threshold_value_fixed), 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            original_cv_bgr = cv2.cvtColor(np.array(original_image.convert('RGB')), cv2.COLOR_RGB2BGR)
            cv2.drawContours(original_cv_bgr, contours, -1, (0, 255, 0), 2) # Green contours

            return Image.fromarray(cv2.cvtColor(original_cv_bgr, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.error(f"Error generating segmentation map: {e}", exc_info=True)
            return original_image.copy()

    def _create_heatmap_overlay(self, heatmap_raw_np: np.ndarray, original_image: Image.Image) -> Image.Image:
        """
        Overlays a focused and visually appealing heatmap onto the original image.
        This function now takes the raw numpy heatmap (not yet normalized for this specific overlay).
        """
        # Ensure heatmap_raw_np is 2D
        if heatmap_raw_np.ndim != 2:
            logger.error(f"heatmap_raw_np for overlay is not 2D, shape: {heatmap_raw_np.shape}. Creating placeholder.")
            return self._create_placeholder_heatmap(original_image) # Fallback to full placeholder logic

        logger.info(f"Heatmap stats for overlay: min={heatmap_raw_np.min():.4f}, max={heatmap_raw_np.max():.4f}, mean={heatmap_raw_np.mean():.4f}")

        threshold_percentile = 95

        # Ensure heatmap_raw_np is not empty and has variation before percentile
        if heatmap_raw_np.size == 0 or heatmap_raw_np.max() == heatmap_raw_np.min():
            threshold_value = heatmap_raw_np.min() # or some other default
        else:
            threshold_value = np.percentile(heatmap_raw_np, threshold_percentile)

        heatmap_for_overlay = np.copy(heatmap_raw_np)
        heatmap_for_overlay[heatmap_for_overlay < threshold_value] = 0

        if heatmap_for_overlay.max() > heatmap_for_overlay.min():
            heatmap_normalized_overlay = (heatmap_for_overlay - heatmap_for_overlay.min()) / (heatmap_for_overlay.max() - heatmap_for_overlay.min())
        else:
            heatmap_normalized_overlay = np.zeros_like(heatmap_for_overlay)

        heatmap_uint8 = np.uint8(255 * heatmap_normalized_overlay)
        
        # Resize heatmap to match the original image size
        heatmap_resized = cv2.resize(heatmap_uint8, original_image.size, interpolation=cv2.INTER_LINEAR)

        # Apply a Gaussian blur to smooth the edges of the heatmap for a more polished look
        heatmap_blurred = cv2.GaussianBlur(heatmap_resized, (25, 25), 0)
        
        # Apply a color map that creates blue-to-red gradient like the reference image
        heatmap_colored = cv2.applyColorMap(heatmap_blurred, cv2.COLORMAP_JET)
        
        # Convert PIL image to OpenCV format
        original_cv = cv2.cvtColor(np.array(original_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        
        # Blend the heatmap with the original image, giving equal weight to both
        superimposed_img = cv2.addWeighted(heatmap_colored, 0.5, original_cv, 0.5, 0)
        
        # Convert back to PIL format
        return Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))

    def _create_placeholder_heatmap(self, original_image: Image.Image) -> Image.Image:
        """
        Create a placeholder heatmap when attention weights are unavailable.
        """
        # Create a subtle center-focused heatmap as placeholder
        h, w = original_image.size[1], original_image.size[0]  # Note: PIL size is (width, height)
        
        # Create a radial gradient from center
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        
        # Calculate distance from center, normalized
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Create a subtle radial attention pattern (higher in center)
        heatmap = 1.0 - (distance / max_distance) * 0.3  # Subtle gradient
        heatmap = np.clip(heatmap, 0.3, 0.7)  # Keep it subtle
        
        return self._create_heatmap_overlay(heatmap, original_image)

    async def analyze_image_from_upload(self, file: UploadFile) -> tuple[AnalysisResponse, Image.Image, Image.Image, Image.Image, Image.Image, Image.Image]:
        """
        Analyze medical image from uploaded file and return complete analysis response with images.
        
        Processes uploaded image files through the complete analysis pipeline,
        including validation, preprocessing, AI analysis, and metadata collection.
        
        Args:
            file: FastAPI UploadFile containing the medical image
            
        Returns:
            tuple: (AnalysisResponse, original_image, ai_attention_heatmap, new_saliency_map, probability_map, segmentation_map)
            - AnalysisResponse: Complete analysis results
            - original_image: Original PIL Image
            - ai_attention_heatmap: AI Attention Heatmap (colored overlay)
            - new_saliency_map: Grayscale Saliency Map
            - probability_map: Grayscale Probability Map
            - segmentation_map: Segmentation Map with contours
            
        Raises:
            HTTPException: If file processing or analysis fails
            
        Process:
        1. File validation and size checking
        2. Image preprocessing and format conversion
        3. Metadata extraction
        4. AI-powered analysis with generation of all maps
        5. Response structuring with unique ID
        """
        try:
            contents = await file.read()
            
            # Validate file size
            if len(contents) > self.max_image_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image too large. Maximum size: {self.max_image_size/1024/1024:.1f}MB"
                )
            
            # Process image and convert to RGB
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            
            # Extract comprehensive metadata
            image_metadata = {
                "source": "upload",
                "filename": file.filename,
                "size": len(contents),
                "format": image.format,
                "mode": image.mode,
                "dimensions": image.size,
                "content_type": file.content_type
            }
            
            # Perform AI analysis
            raw_response, findings, ai_attention_heatmap, new_saliency_map, probability_map, segmentation_map = await self.analyze_image(image)
            
            # Generate unique analysis ID and timestamp
            analysis_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            analysis_response = AnalysisResponse(
                analysis_id=analysis_id,
                timestamp=timestamp,
                findings=findings,
                raw_response=raw_response,
                image_metadata=image_metadata
            )
            
            return analysis_response, image, ai_attention_heatmap, new_saliency_map, probability_map, segmentation_map
            
        except Exception as e:
            logger.error(f"Error analyzing uploaded image: {str(e)}")
            if isinstance(e, HTTPException):            raise
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    async def analyze_image_from_url(self, image_url: str) -> tuple[AnalysisResponse, Image.Image, Image.Image, Image.Image, Image.Image, Image.Image]:
        """
        Analyze medical image from URL and return complete analysis response with images.
        
        Downloads and analyzes medical images from HTTP/HTTPS URLs with comprehensive
        validation, security checks, and metadata collection.
        
        Args:
            image_url: Valid HTTP/HTTPS URL pointing to a medical image

        Returns:
            tuple: (AnalysisResponse, original_image, ai_attention_heatmap, new_saliency_map, probability_map, segmentation_map)
            - AnalysisResponse: Complete analysis results
            - original_image: Original PIL Image
            - ai_attention_heatmap: AI Attention Heatmap (colored overlay)
            - new_saliency_map: Grayscale Saliency Map
            - probability_map: Grayscale Probability Map
            - segmentation_map: Segmentation Map with contours
            
        Raises:
            HTTPException: If URL fetch or analysis fails
            
        Security Features:
        - URL validation and format checking
        - Protocol restriction (HTTP/HTTPS only)
        - Content-Type verification
        - File size limitations
        - Timeout protection
        - Comprehensive error handling
        """
        try:
            # Fetch image with security validation
            image, image_metadata = await self.fetch_image_from_url(image_url)
            
            # Perform AI analysis
            raw_response, findings, ai_attention_heatmap, new_saliency_map, probability_map, segmentation_map = await self.analyze_image(image)
            
            # Generate unique analysis ID and timestamp
            analysis_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Update metadata with URL source info
            image_metadata.update({
                "source": "url",
                "url": image_url
            })
            
            analysis_response = AnalysisResponse(
                analysis_id=analysis_id,
                timestamp=timestamp,
                findings=findings,
                raw_response=raw_response,
                image_metadata=image_metadata
            )
            
            return analysis_response, image, ai_attention_heatmap, new_saliency_map, probability_map, segmentation_map
            
        except Exception as e:
            logger.error(f"Error analyzing image from URL: {str(e)}")
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    async def fetch_image_from_url(self, image_url: str) -> tuple[Image.Image, Dict[str, Any]]:
        """
        Securely fetch medical image from HTTP/HTTPS URL with comprehensive validation.
        
        Downloads images from URLs with robust security measures, format validation,
        and metadata extraction suitable for medical image processing.
        
        Args:
            image_url: HTTP/HTTPS URL pointing to an image resource
            
        Returns:
            tuple: (PIL_Image, metadata_dict)
            - PIL_Image: Processed image in RGB format
            - metadata_dict: Comprehensive image and download metadata
            
        Raises:
            HTTPException: For various failure scenarios with specific error codes
            
        Security Measures:
        - URL format validation
        - Protocol restriction (HTTP/HTTPS only)
        - Content-Type verification
        - File size limitations (pre and post download)
        - Timeout protection (30 seconds)
        - Proper User-Agent identification
        
        Performance Optimizations:
        - Efficient async HTTP handling
        - Memory-conscious image processing
        - Comprehensive error categorization
        """
        try:
            # URL validation and security checks
            if not validators.url(image_url):
                raise HTTPException(status_code=400, detail="Invalid URL format")
            if not image_url.startswith(('https://', 'http://')):
                raise HTTPException(status_code=400, detail="Only HTTP/HTTPS URLs are supported")
            
            # Configure secure HTTP client
            headers = {
                'User-Agent': 'Medical-Image-Analysis-API/1.0.0',
                'Accept': 'image/*'
            }
            timeout = aiohttp.ClientTimeout(total=30)
            
            # Secure image download with validation
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(image_url, headers=headers) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to fetch image. HTTP {response.status}: {response.reason}"
                        )
                    
                    # Content validation
                    content_type = response.headers.get('content-type', '').lower()
                    if not content_type.startswith('image/'):
                        raise HTTPException(
                            status_code=400,
                            detail=f"URL does not point to an image. Content-Type: {content_type}"
                        )
                    
                    # Size validation (pre-download)
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > self.max_image_size:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Image too large. Maximum size: {self.max_image_size/1024/1024:.1f}MB"
                        )
                    
                    # Download and validate actual size
                    image_data = await response.read()
                    if len(image_data) > self.max_image_size:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Image too large. Maximum size: {self.max_image_size/1024/1024:.1f}MB"
                        )
            
            # Image processing and validation
            try:
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
            
            # Comprehensive metadata collection
            image_metadata = {
                "source": "url",
                "url": image_url,
                "size": len(image_data),
                "format": image.format,
                "mode": image.mode,
                "dimensions": image.size,
                "content_type": content_type
            }
            
            logger.info(f"Successfully fetched image from URL: {image_url}")
            return image, image_metadata
            
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching image from {image_url}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Network error: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching image from {image_url}: {str(e)}")
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=f"Failed to fetch image: {str(e)}")

# Initialize analyzer
analyzer = MedicalImageAnalyzer()
# Initialize database
initialize_database()


@app.get("/")
async def root():
    """
    Health check endpoint for API status verification.
    
    Returns:
        dict: API status and basic information
    """
    return {"message": "Medical Image Analysis API (Med-GEMMA-3) is running", "status": "healthy"}


@app.post("/analyze-and-report", response_model=PDFGenerationResponse)
async def analyze_and_generate_report(
    file: UploadFile = File(...)
):
    """
    Analyze medical image and generate comprehensive PDF report.
    
    Combines image analysis with professional PDF report generation
    optimized for medical documentation and record keeping.
    
    Args:
        file: Uploaded medical image file
        
    Returns:
        PDFGenerationResponse: PDF file path and analysis identifier
        
    Performance Optimization:
        - Uses thread pool for CPU-intensive PDF generation
        - Non-blocking async execution
        - Memory-efficient processing
        - Single AI inference run (no duplicate processing)
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Perform analysis once - this includes heatmap generation
    analysis_response, original_image, heatmap_image = await analyzer.analyze_image_from_upload(file)
    
    # Generate PDF in thread pool to avoid blocking the event loop
    pdf_path = await run_in_threadpool(
        analyzer.generate_pdf_report, 
        analysis_response,
        original_image,
        heatmap_image
    )
    
    return PDFGenerationResponse(
        pdf_path=pdf_path,
        analysis_id=analysis_response.analysis_id
    )

@app.get("/reports")
async def list_reports():
    """
    List all generated medical reports with metadata.
    
    Returns:
        dict: List of reports with filenames, IDs, timestamps, and sizes
        
    Raises:
        HTTPException: If listing operation fails
    """
    try:
        reports = []
        if os.path.exists(analyzer.reports_dir):
            for filename in os.listdir(analyzer.reports_dir):
                if filename.endswith('.pdf'):
                    filepath = os.path.join(analyzer.reports_dir, filename)
                    stats = os.stat(filepath)
                    reports.append({
                        "filename": filename,
                        "analysis_id": filename.replace("medical_report_", "").replace(".pdf", ""),
                        "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat(),
                        "size": stats.st_size
                    })
        return {"reports": reports}
    except Exception as e:
        logger.error(f"Error listing reports: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list reports")


@app.post("/analyze-url", response_model=AnalysisResponse)
async def analyze_image_url(request: ImageURLRequest):
    """
    Analyze medical image from HTTP/HTTPS URL using Med-GEMMA-3.
    
    Downloads and analyzes medical images from remote URLs with comprehensive
    security validation and error handling.
    
    Args:
        request: ImageURLRequest containing the image URL
        
    Returns:
        AnalysisResponse: Complete analysis results with structured findings
        
    Security Features:
        - URL format validation using Pydantic HttpUrl
        - Content-Type verification
        - File size restrictions
        - Timeout protection
        - Protocol validation (HTTP/HTTPS only)
    """
    analysis_response, _, _ = await analyzer.analyze_image_from_url(str(request.image_url))
    return analysis_response

# The working path for the PDF report generation
@app.post("/analyze-url-and-report", response_model=PDFGenerationResponse)
async def analyze_url_and_generate_report(
    request: ImageURLReportRequest
):
    """
    Analyze medical image from URL and generate comprehensive PDF report.
    
    Combines secure image downloading, AI analysis, professional PDF
    report generation, S3 upload, and database logging.
    
    Args:
        request: ImageURLReportRequest containing the image URL
        
    Returns:
        PDFGenerationResponse: PDF file S3 path and analysis identifier
        
    Performance Optimization:
        - Uses thread pool for CPU-intensive PDF generation
        - Efficient async image downloading
        - Memory-conscious processing pipeline
        - Single AI inference run (no duplicate processing)
    """
    # Perform analysis once - this includes heatmap generation
    analysis_response, original_image, heatmap_image, saliency_map, probability_map, segmentation_map  = await analyzer.analyze_image_from_url(str(request.image_url))
    
    # Generate PDF in thread pool to avoid blocking the event loop
    local_pdf_path = await run_in_threadpool(
        pdf_generator.generate_pdf_report, 
        analysis_response,
        original_image,
        heatmap_image,
        saliency_map,
        probability_map,
        segmentation_map
    )
    
    # Define S3 bucket details (replace with your actual configuration)
    s3_bucket_name = os.getenv("AWS_S3_BUCKET_NAME", "your-s3-bucket-name")
    s3_region = os.getenv("AWS_REGION", "your-aws-region")
    # Create a unique S3 key, e.g., using the analysis ID
    s3_key = f"reports/{analysis_response.analysis_id}/{os.path.basename(local_pdf_path)}"
    
    s3_pdf_url = ""  # Initialize to empty string
    try:
        # Upload the PDF to S3
        s3_pdf_url = await upload_to_s3(
            local_file_path=local_pdf_path,
            bucket_name=s3_bucket_name,
            s3_key=s3_key,
            region_name=s3_region
        )
        logger.info(f"Successfully uploaded PDF to S3: {s3_pdf_url}")
    except Exception as e:
        logger.error(f"Failed to upload PDF to S3: {e}", exc_info=True)

    path_to_store = s3_key
    
    await store_analysis_result(
        image_uuid=request.image_uuid,  # Ensure ImageURLReportRequest has image_uuid
        reportPath=path_to_store
    )
    
    return PDFGenerationResponse(
        pdf_path=path_to_store,  # Return S3 URL or local path
        analysis_id=analysis_response.analysis_id
    )


@app.get("/model-status", response_model=ModelStatus)
async def get_model_status():
    """
    Get current Med-GEMMA-3 model loading status and system information.
    
    Provides detailed information about model state, device configuration,
    memory usage, and any loading errors for monitoring and debugging.
    
    Returns:
        ModelStatus: Model state, device info, memory usage, and error details
        
    Monitoring Features:
        - Model loading state
        - GPU/CPU device information
        - Memory usage statistics
        - Error tracking and reporting
    """
    global model_loading_error
    
    try:
        if analyzer.model is not None and model_loading_error is None:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            memory_info = analyzer.get_system_memory_info()
            return ModelStatus(
                loaded=True,
                device=device_str,
                memory_usage=memory_info,
                error=None
            )
        elif model_loading_error is not None:
            return ModelStatus(
                loaded=False,
                device="unknown",
                memory_usage=analyzer.get_system_memory_info(),
                error=model_loading_error
            )
        else:
            return ModelStatus(
                loaded=False,
                device="unknown",
                memory_usage=analyzer.get_system_memory_info(),
                error="Model not yet loaded"
            )
    except Exception as e:
        return ModelStatus(
            loaded=False,
            device="unknown",
            memory_usage="unavailable",
            error=str(e)
        )


@app.post("/load-model")
async def force_load_model():
    """
    Force reload the Med-GEMMA-3 pipeline for recovery from loading failures.
    
    Useful for recovering from initial loading failures or reinitializing
    the model after configuration changes.
    
    Returns:
        dict: Loading status and result message
        
    Recovery Features:
        - Clears previous error states
        - Reinitializes model pipeline
        - Comprehensive error reporting
        - Memory cleanup and optimization
    """
    global model_loading_error
    model_loading_error = None
    
    # Reset pipeline for clean reload
    analyzer.model = None
    analyzer.processor = None
    analyzer.model_loaded = False
    
    try:
        await analyzer.load_model()
        return {"message": "Model loading initiated", "status": "success"}
    except Exception as e:
        return {"message": f"Model loading failed: {str(e)}", "status": "error"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
