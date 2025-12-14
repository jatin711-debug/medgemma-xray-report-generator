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

# Global analyzer instances
analyzer = None  # MedGemma analyzer (legacy)
multi_analyzer = None  # Multi-model analyzer (primary)

from db import initialize_database, store_analysis_result
from aws_utils import upload_to_s3 # Import the new S3 upload function
from multi_model_analyzer import MultiModelAnalyzer, ConsensusFindings, MultiModelAnalysisResponse
from nv_reason_analyzer import NVReasonFindings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Medical Image Analysis API (Multi-Model: MedGemma + NV-Reason-CXR)...")
    global analyzer, multi_analyzer
    
    # Initialize MedGemma analyzer
    analyzer = MedicalImageAnalyzer()
    await analyzer.load_model()
    
    # Initialize Multi-Model analyzer (uses NV-Reason-CXR for heatmaps)
    multi_analyzer = MultiModelAnalyzer()
    multi_analyzer.set_medgemma_analyzer(analyzer)  # Share MedGemma instance
    await multi_analyzer.load_models()
    
    logger.info("All models loaded successfully!")
    yield
    # Shutdown - cleanup if needed
    logger.info("Shutting down Medical Image Analysis API...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Medical Image Analysis API (Multi-Model)",
    description="AI-powered medical image analysis using MedGemma-4B-IT and NV-Reason-CXR-3B with consensus-based findings",
    version="2.0.0",
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
    
    def parse_ai_response_multi(self, raw_response: str) -> RadiologyFindings:
        """
        Parse a structured AI model response for multiple conditions.
        Fixes the bug where "no pneumothorax detected" was misinterpreted.
        
        Args:
            raw_response: Raw text output from the Med-GEMMA model.

        Returns:
            RadiologyFindings: Structured medical findings for all analyzed conditions.
        """
        try:
            findings = {}
            # Regex to capture each condition block
            pattern = re.compile(r"\*\*Finding: (Pneumothorax|Pneumonia|TB)\*\*\n- \*\*Status\*\*: (Detected|Not Detected)\n- \*\*Confidence\*\*: ([\d.]+%|N/A)", re.IGNORECASE)
            matches = pattern.findall(raw_response)

            for match in matches:
                condition, status, confidence = match
                findings[condition.strip()] = {
                    "status": status.strip(),
                    "confidence": confidence.strip()
                }

            if not findings:
                # Fallback for unstructured responses
                logger.warning("Could not parse structured response, using fallback.")
                return RadiologyFindings(
                    anatomical_structures=["Chest structures"],
                    abnormalities=["Could not parse AI response."],
                    impression="Inconclusive",
                    recommendations=["Manual review required."],
                    confidence_level="N/A"
                )

            detected_conditions = [f"{cond} ({f['confidence']})" for cond, f in findings.items() if f['status'].lower() == 'detected']

            if detected_conditions:
                impression = "Positive for " + ", ".join(c.split(' ')[0] for c in detected_conditions)
                abnormalities = detected_conditions
            else:
                impression = "Negative for all tested conditions"
                abnormalities = ["No evidence of Pneumothorax, Pneumonia, or TB found."]

            # Use the highest confidence among detected conditions, or "N/A"
            confidences = [float(f['confidence'].replace('%', '')) for f in findings.values() if '%' in f['confidence']]
            overall_confidence = f"{max(confidences):.1f}%" if confidences else "N/A"

            return RadiologyFindings(
                anatomical_structures=["Lungs", "Pleura", "Mediastinum"],
                abnormalities=abnormalities,
                impression=impression,
                recommendations=["Clinical correlation and expert review is recommended."],
                confidence_level=overall_confidence
            )

        except Exception as e:
            logger.error(f"Error parsing multi-condition AI response: {str(e)}")
            return RadiologyFindings(
                anatomical_structures=["Analysis completed"],
                abnormalities=["Unable to parse detailed findings"],
                impression=raw_response,
                recommendations=["Manual review recommended"],
                confidence_level="Not available"
            )

    async def analyze_image(self, image: Image.Image) -> tuple[str, RadiologyFindings, Image.Image]:
        """
        Perform AI-powered medical image analysis using Med-GEMMA.
        Returns raw text response, structured findings, and AI Attention Heatmap.
        """
        await self.load_model()

        try:
            # Resize image to match model input expectations (usually 224x224)
            image_for_model = image.resize((224, 224))

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a highly experienced radiologist. Analyze chest X-rays for Pneumothorax, Pneumonia, and TB."}]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Review this chest X-ray for signs of Pneumothorax, Pneumonia, and Tuberculosis (TB).\n"
                                "Respond strictly in the following format for EACH condition:\n\n"
                                "**Finding: Pneumothorax**\n"
                                "- **Status**: [Detected | Not Detected]\n"
                                "- **Confidence**: [XX%]\n\n"
                                "**Finding: Pneumonia**\n"
                                "- **Status**: [Detected | Not Detected]\n"
                                "- **Confidence**: [XX%]\n\n"
                                "**Finding: TB**\n"
                                "- **Status**: [Detected | Not Detected]\n"
                                "- **Confidence**: [XX%]\n\n"
                                "Do not include any reasoning, disclaimers, or extra comments. Provide only the structured findings."
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

            # Generate heatmap
            heatmap_image, _ = self._generate_heatmap_and_raw_normalized(attentions, image)

            findings = self.parse_ai_response_multi(raw_response)
            return raw_response, findings, heatmap_image

        except Exception as e:
            logger.error(f"Error during image analysis: {str(e)}", exc_info=True)
            # Return placeholders if analysis fails critically
            placeholder_img = self._create_placeholder_heatmap(image)
            findings = RadiologyFindings(
                anatomical_structures=["Error during analysis"],
                abnormalities=[f"Error: {str(e)}"],
                impression="Analysis Incomplete",
                recommendations=["Review error logs"],
                confidence_level="N/A"
            )
            return f"Error: {str(e)}", findings, placeholder_img


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
        """
        Generates a saliency map by modulating the intensity of a grayscale version
        of the original image with normalized heatmap values.

        This method highlights regions of interest (where heatmap values are high)
        while preserving the underlying structure of the original image in grayscale.

        Args:
            heatmap_normalized_np: A 2D NumPy array representing the normalized attention
                                   heatmap (ideally values between 0 and 1). If None or
                                   invalid, a grayscale version of the original image
                                   is returned as a fallback.
            original_image: A PIL.Image.Image object, the original image.

        Returns:
            A PIL.Image.Image object representing the saliency map (RGB).
            Returns a grayscale version of original_image if inputs are invalid or
            an error occurs.
        """
        target_size = original_image.size
        fallback_image = original_image.convert("L").convert("RGB")

        if not isinstance(original_image, Image.Image):
            logger.error("Invalid input: original_image is not a PIL Image. Returning grayscale original.")
            return fallback_image

        if heatmap_normalized_np is None:
            logger.warning("heatmap_normalized_np is None for new saliency map. Returning grayscale original.")
            return fallback_image

        if not isinstance(heatmap_normalized_np, np.ndarray):
            logger.warning(f"Invalid input: heatmap_normalized_np is not a NumPy array (type: {type(heatmap_normalized_np)}). Returning grayscale original.")
            return fallback_image

        if heatmap_normalized_np.ndim != 2:
            logger.warning(f"Invalid input: heatmap_normalized_np is not 2D (shape: {heatmap_normalized_np.shape}). Returning grayscale original.")
            return fallback_image

        if heatmap_normalized_np.size == 0:
            logger.warning("Invalid input: heatmap_normalized_np is empty. Returning grayscale original.")
            return fallback_image

        try:
            # Defensive copy, as heatmap_normalized_np might be used elsewhere.
            heatmap_to_process = np.copy(heatmap_normalized_np)

            # Ensure heatmap values are robustly scaled to 0-1 if they aren't already
            # This handles cases where input might not be perfectly normalized.
            min_val, max_val = heatmap_to_process.min(), heatmap_to_process.max()
            if min_val == max_val: # Avoid division by zero if flat
                heatmap_scaled = np.zeros_like(heatmap_to_process) if min_val == 0 else np.ones_like(heatmap_to_process) * 0.5 # Default to mid-gray or black
            else:
                heatmap_scaled = (heatmap_to_process - min_val) / (max_val - min_val)

            logger.info(f"Saliency map input heatmap stats (after potential rescale): min={heatmap_scaled.min():.4f}, max={heatmap_scaled.max():.4f}, mean={heatmap_scaled.mean():.4f}")

            # Resize heatmap to match the original image size
            # Using INTER_LINEAR for a balance of speed and quality.
            heatmap_resized = cv2.resize(heatmap_scaled, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Convert original image to grayscale
            original_gray = original_image.convert("L")
            original_gray_np = np.array(original_gray, dtype=np.float32) # Use float32 for multiplication
            
            # Create saliency map by multiplying grayscale image with attention weights
            # This highlights important regions while preserving image structure.
            saliency_map_float = original_gray_np * heatmap_resized
            
            # Normalize the resulting saliency map to 0-255 range for image display
            saliency_map_norm_float = cv2.normalize(saliency_map_float, None, 0, 255, cv2.NORM_MINMAX)
            saliency_map_uint8 = saliency_map_norm_float.astype(np.uint8)
            
            saliency_pil_image = Image.fromarray(saliency_map_uint8).convert("RGB")
            logger.info("Successfully generated new saliency map.")
            return saliency_pil_image
            
        except Exception as e:
            logger.error(f"Error generating new saliency map: {e}", exc_info=True)
            return fallback_image

    def _generate_probability_map(self, heatmap_normalized_np: Optional[np.ndarray], original_image: Image.Image) -> Image.Image:
        """
        Generates a probability map visualizing normalized heatmap values as grayscale intensities.

        This map directly represents the (resized) heatmap, where brighter areas indicate
        higher values in the heatmap_normalized_np.

        Args:
            heatmap_normalized_np: A 2D NumPy array representing the normalized attention
                                   heatmap (ideally values between 0 and 1). If None or
                                   invalid, a grayscale version of the original image
                                   is returned as a fallback.
            original_image: A PIL.Image.Image object, used for determining target size and
                            as a base for the fallback image.

        Returns:
            A PIL.Image.Image object representing the probability map (RGB, grayscale content).
            Returns a grayscale version of original_image if inputs are invalid or
            an error occurs.
        """
        target_size = original_image.size
        # Fallback is a grayscale version of the original image, converted to RGB for consistency.
        fallback_image = original_image.convert("L").convert("RGB")

        if not isinstance(original_image, Image.Image):
            logger.error("Invalid input: original_image is not a PIL Image. Returning grayscale original.")
            return fallback_image

        if heatmap_normalized_np is None:
            logger.warning("heatmap_normalized_np is None for probability map. Returning grayscale original.")
            return fallback_image

        if not isinstance(heatmap_normalized_np, np.ndarray):
            logger.warning(f"Invalid input: heatmap_normalized_np is not a NumPy array (type: {type(heatmap_normalized_np)}). Returning grayscale original.")
            return fallback_image

        if heatmap_normalized_np.ndim != 2:
            logger.warning(f"Invalid input: heatmap_normalized_np is not 2D (shape: {heatmap_normalized_np.shape}). Returning grayscale original.")
            return fallback_image

        if heatmap_normalized_np.size == 0:
            logger.warning("Invalid input: heatmap_normalized_np is empty. Returning grayscale original.")
            return fallback_image

        try:
            # Defensive copy
            heatmap_to_process = np.copy(heatmap_normalized_np)

            # Ensure heatmap values are robustly scaled to 0-1.
            min_val, max_val = heatmap_to_process.min(), heatmap_to_process.max()
            if min_val == max_val:
                heatmap_scaled = np.zeros_like(heatmap_to_process) if min_val == 0 else np.ones_like(heatmap_to_process) * 0.5
            else:
                heatmap_scaled = (heatmap_to_process - min_val) / (max_val - min_val)

            logger.info(f"Probability map input heatmap stats (after potential rescale): min={heatmap_scaled.min():.4f}, max={heatmap_scaled.max():.4f}, mean={heatmap_scaled.mean():.4f}")

            # Resize heatmap to match the original image size
            heatmap_resized = cv2.resize(heatmap_scaled, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Convert to uint8 (0-255 range) - this represents probability/attention strength
            # Values are directly scaled from the 0-1 heatmap_resized.
            prob_map_uint8 = np.uint8(255 * heatmap_resized)
            
            # Convert to RGB PIL Image (though content is grayscale)
            prob_map_pil = Image.fromarray(prob_map_uint8).convert("RGB")
            logger.info("Successfully generated probability map.")
            return prob_map_pil
            
        except Exception as e:
            logger.error(f"Error generating probability map: {e}", exc_info=True)
            return fallback_image

    def _generate_segmentation_map(self, heatmap_normalized_np: Optional[np.ndarray], original_image: Image.Image) -> Image.Image:
        """
        Generates a segmentation map by thresholding the normalized heatmap and
        drawing contours of the thresholded regions onto the original image.

        Args:
            heatmap_normalized_np: A 2D NumPy array representing the normalized attention
                                   heatmap (ideally values between 0 and 1). If None or
                                   invalid, a copy of the original image is returned.
            original_image: A PIL.Image.Image object, which will be the base for
                            drawing contours.

        Returns:
            A PIL.Image.Image object with contours drawn on it (RGB).
            Returns a copy of original_image if inputs are invalid or an error occurs.
        """
        # Fallback is a copy of the original image.
        fallback_image = original_image.copy()

        if not isinstance(original_image, Image.Image):
            logger.error("Invalid input: original_image is not a PIL Image. Returning copy of original.")
            return fallback_image

        if heatmap_normalized_np is None:
            logger.warning("heatmap_normalized_np is None for segmentation map. Returning copy of original image.")
            return fallback_image

        if not isinstance(heatmap_normalized_np, np.ndarray):
            logger.warning(f"Invalid input: heatmap_normalized_np is not a NumPy array (type: {type(heatmap_normalized_np)}). Returning copy of original.")
            return fallback_image

        if heatmap_normalized_np.ndim != 2:
            logger.warning(f"Invalid input: heatmap_normalized_np is not 2D (shape: {heatmap_normalized_np.shape}). Returning copy of original.")
            return fallback_image

        if heatmap_normalized_np.size == 0:
            logger.warning("Invalid input: heatmap_normalized_np is empty. Returning copy of original.")
            return fallback_image

        try:
            # Defensive copy of the heatmap
            heatmap_to_process = np.copy(heatmap_normalized_np)

            # Ensure heatmap values are robustly scaled to 0-1.
            min_val, max_val = heatmap_to_process.min(), heatmap_to_process.max()
            if min_val == max_val: # If flat, Otsu might not work well or give unexpected results.
                                   # Depending on the value, it might become all black or all white.
                heatmap_scaled = np.zeros_like(heatmap_to_process) if min_val < 0.5 else np.ones_like(heatmap_to_process)
                logger.warning(f"Segmentation map input heatmap is flat (all values {min_val}). Resulting mask might be trivial.")
            else:
                heatmap_scaled = (heatmap_to_process - min_val) / (max_val - min_val)

            logger.info(f"Segmentation map input heatmap stats (after potential rescale): min={heatmap_scaled.min():.4f}, max={heatmap_scaled.max():.4f}, mean={heatmap_scaled.mean():.4f}")

            # Resize heatmap to match the original image size for accurate segmentation
            heatmap_resized = cv2.resize(heatmap_scaled, original_image.size, interpolation=cv2.INTER_LINEAR)

            # Convert the 0-1 scaled heatmap to 0-255 uint8 for thresholding
            threshold_input_uint8 = np.uint8(255 * heatmap_resized)

            # Apply Otsu's thresholding to binarize the image.
            # Otsu's method automatically determines the optimal threshold value.
            threshold_value, binary_mask = cv2.threshold(
                threshold_input_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            logger.info(f"Otsu's thresholding applied for segmentation map. Determined threshold value: {threshold_value}")

            # Find contours in the binary mask.
            # RETR_EXTERNAL retrieves only the external contours.
            # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments, leaving only their end points.
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            logger.info(f"Found {len(contours)} contours for segmentation map.")

            if not contours:
                logger.info("No contours found. Returning original image with no overlays.")
                return fallback_image # Or original_image.copy() if no modification desired

            # Prepare the original image for drawing contours.
            # Convert PIL image to OpenCV format (BGR). A copy is implicitly made by np.array.
            original_cv_bgr = cv2.cvtColor(np.array(original_image.convert('RGB')), cv2.COLOR_RGB2BGR)

            # Draw contours on the image.
            # Parameters: image, contours, contourIdx (-1 for all), color (Green), thickness (2)
            cv2.drawContours(original_cv_bgr, contours, -1, (0, 255, 0), 2)

            # Convert the image back to PIL format (RGB).
            segmented_image_pil = Image.fromarray(cv2.cvtColor(original_cv_bgr, cv2.COLOR_BGR2RGB))
            logger.info("Successfully generated segmentation map with contours.")
            return segmented_image_pil

        except Exception as e:
            logger.error(f"Error generating segmentation map: {e}", exc_info=True)
            return fallback_image # Return a copy of the original image on error

    # --- Constants for heatmap overlay ---
    _HEATMAP_THRESHOLD_PERCENTILE = 95
    _GAUSSIAN_BLUR_KERNEL_SIZE = (25, 25) # Must be odd numbers
    _HEATMAP_OVERLAY_ALPHA = 0.5 # Blending factor for the heatmap

    def _create_heatmap_overlay(self, heatmap_raw_np: Optional[np.ndarray], original_image: Image.Image, is_placeholder: bool = False) -> Image.Image:
        """
        Overlays a focused and visually appealing heatmap onto the original image.

        Args:
            heatmap_raw_np: The raw 2D NumPy array heatmap. Expected to be roughly normalized (e.g. 0-1)
                            if it's a real heatmap, or a specific pattern if it's a placeholder.
                            If None or invalid, a placeholder is generated.
            original_image: The PIL.Image.Image object to overlay the heatmap on.
            is_placeholder: If True, skips the percentile thresholding to ensure the
                            placeholder pattern is fully visible.

        Returns:
            A PIL.Image.Image object with the heatmap overlay.
        """
        if not isinstance(original_image, Image.Image):
            logger.error("Invalid original_image provided to _create_heatmap_overlay. Cannot proceed.")
            # Cannot create placeholder without a valid original_image for size.
            # Return a small, simple, fixed-size gray image as an ultimate fallback.
            return Image.new('RGB', (224, 224), color='gray')

        if heatmap_raw_np is None or not isinstance(heatmap_raw_np, np.ndarray) or heatmap_raw_np.ndim != 2 or heatmap_raw_np.size == 0:
            logger.warning(f"Invalid heatmap_raw_np for overlay (is_placeholder={is_placeholder}). Type: {type(heatmap_raw_np)}. Shape: {getattr(heatmap_raw_np, 'shape', 'N/A')}. Generating placeholder.")
            # Call _generate_raw_placeholder_pattern to get the pattern, then re-call self with is_placeholder=True
            placeholder_pattern = self._generate_raw_placeholder_pattern(original_image)
            return self._create_heatmap_overlay(placeholder_pattern, original_image, is_placeholder=True)

        try:
            logger.info(f"Creating heatmap overlay. Input stats: min={heatmap_raw_np.min():.4f}, max={heatmap_raw_np.max():.4f}, mean={heatmap_raw_np.mean():.4f}. Is placeholder: {is_placeholder}")

            heatmap_processed = np.copy(heatmap_raw_np)

            if not is_placeholder:
                # Apply percentile thresholding only for actual heatmaps to focus on salient areas
                if heatmap_processed.max() > heatmap_processed.min(): # Avoid percentile on flat heatmaps
                    threshold_value = np.percentile(heatmap_processed, self._HEATMAP_THRESHOLD_PERCENTILE)
                    heatmap_processed[heatmap_processed < threshold_value] = 0
                    logger.info(f"Applied {self._HEATMAP_THRESHOLD_PERCENTILE}th percentile threshold: {threshold_value:.4f}")
                else:
                    logger.info("Heatmap is flat, skipping percentile thresholding.")

            # Normalize the processed heatmap (either thresholded or placeholder pattern) to 0-1 range
            min_val, max_val = heatmap_processed.min(), heatmap_processed.max()
            if max_val > min_val:
                heatmap_normalized = (heatmap_processed - min_val) / (max_val - min_val)
            else: # Flat heatmap (all zeros after thresholding, or a flat placeholder)
                heatmap_normalized = np.zeros_like(heatmap_processed)

            heatmap_uint8 = np.uint8(255 * heatmap_normalized)

            heatmap_resized = cv2.resize(heatmap_uint8, original_image.size, interpolation=cv2.INTER_LINEAR)

            # Blur slightly to smooth the heatmap
            # For placeholders, a smaller blur might be better or none, but fine for now.
            blur_kernel = self._GAUSSIAN_BLUR_KERNEL_SIZE
            if is_placeholder and blur_kernel[0] > 5: # Reduce blur for placeholders if it's large
                blur_kernel = (max(1, blur_kernel[0]//2)*2+1, max(1, blur_kernel[1]//2)*2+1) # Ensure odd

            heatmap_blurred = cv2.GaussianBlur(heatmap_resized, blur_kernel, 0)

            heatmap_colored = cv2.applyColorMap(heatmap_blurred, cv2.COLORMAP_JET)

            original_cv_bgr = cv2.cvtColor(np.array(original_image.convert('RGB')), cv2.COLOR_RGB2BGR)

            superimposed_img = cv2.addWeighted(heatmap_colored, self._HEATMAP_OVERLAY_ALPHA, original_cv_bgr, 1.0 - self._HEATMAP_OVERLAY_ALPHA, 0)

            final_image = Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
            logger.info(f"Successfully created heatmap overlay (is_placeholder={is_placeholder}).")
            return final_image

        except Exception as e:
            logger.error(f"Error in _create_heatmap_overlay (is_placeholder={is_placeholder}): {e}", exc_info=True)
            # If overlay creation fails, attempt to return a placeholder
            # To prevent recursion if this itself was called for a placeholder that failed, check is_placeholder
            if not is_placeholder:
                 placeholder_pattern = self._generate_raw_placeholder_pattern(original_image)
                 return self._create_heatmap_overlay(placeholder_pattern, original_image, is_placeholder=True)
            else:
                # Ultimate fallback if creating even a placeholder overlay fails
                logger.error("Critical error: Failed to create even a placeholder overlay. Returning plain gray image.")
                return Image.new('RGB', original_image.size, color='gray')


    def _generate_raw_placeholder_pattern(self, original_image: Image.Image) -> np.ndarray:
        """Generates a raw numpy array pattern for placeholder heatmaps."""
        h, w = original_image.size[1], original_image.size[0]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        
        # Create a more distinct pattern, e.g., a cross or checkerboard, or simpler gradient
        # Radial gradient, but ensure it has enough range to be visible
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        if max_dist == 0: max_dist = 1 # Avoid division by zero for tiny images

        pattern = (distance / max_dist) # Normalized distance, 0 at center, 1 at corners
        pattern = 1.0 - pattern # Invert: 1 at center, 0 at corners
        # Ensure it's quite visible by scaling its range, e.g., from 0.2 to 0.8
        pattern = 0.2 + pattern * 0.6
        return np.clip(pattern, 0.0, 1.0)


    def _create_placeholder_heatmap(self, original_image: Image.Image) -> Image.Image:
        """
        Creates a visually distinct placeholder heatmap overlay.

        This function is called when actual attention weights are unavailable or
        if heatmap generation fails. It generates a predefined pattern and
        overlays it onto the original image.

        Args:
            original_image: The PIL.Image.Image object to overlay the placeholder on.

        Returns:
            A PIL.Image.Image object with the placeholder heatmap overlay.
            Returns a simple gray image of original_image's size if placeholder
            generation itself fails.
        """
        logger.info("Creating a placeholder heatmap.")
        if not isinstance(original_image, Image.Image):
            logger.error("Invalid original_image provided to _create_placeholder_heatmap. Returning fixed-size gray image.")
            return Image.new('RGB', (224, 224), color='lightgray') # Ultimate fallback

        try:
            # Generate the raw numpy pattern for the placeholder
            placeholder_pattern_np = self._generate_raw_placeholder_pattern(original_image)

            # Create the overlay using the main overlay function, flagging it as a placeholder
            return self._create_heatmap_overlay(placeholder_pattern_np, original_image, is_placeholder=True)

        except Exception as e:
            logger.error(f"Failed to create placeholder heatmap: {e}", exc_info=True)
            # If all else fails, return a simple, identifiable image
            try:
                error_img = original_image.copy()
                from PIL import ImageDraw
                draw = ImageDraw.Draw(error_img)
                text = "Placeholder Error"
                # Basic text drawing, might need font for better appearance
                draw.text((10, 10), text, fill=(255,0,0))
                return error_img
            except: # Final fallback if drawing text fails
                 return Image.new('RGB', original_image.size, color='darkgray')


    async def analyze_image_from_upload(self, file: UploadFile) -> tuple[AnalysisResponse, Image.Image, Image.Image]:
        """
        Analyze medical image from uploaded file and return analysis response with heatmap.
        
        Processes uploaded image files through the complete analysis pipeline,
        including validation, preprocessing, AI analysis, and metadata collection.
        
        Args:
            file: FastAPI UploadFile containing the medical image
            
        Returns:
            tuple: (AnalysisResponse, original_image, ai_attention_heatmap)
            - AnalysisResponse: Complete analysis results
            - original_image: Original PIL Image
            - ai_attention_heatmap: AI Attention Heatmap (colored overlay)
            
        Raises:
            HTTPException: If file processing or analysis fails
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
            raw_response, findings, ai_attention_heatmap = await self.analyze_image(image)
            
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
            
            return analysis_response, image, ai_attention_heatmap
            
        except Exception as e:
            logger.error(f"Error analyzing uploaded image: {str(e)}")
            if isinstance(e, HTTPException):            raise
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    async def analyze_image_from_url(self, image_url: str) -> tuple[AnalysisResponse, Image.Image, Image.Image]:
        """
        Analyze medical image from URL and return analysis response with heatmap.
        
        Downloads and analyzes medical images from HTTP/HTTPS URLs with comprehensive
        validation, security checks, and metadata collection.
        
        Args:
            image_url: Valid HTTP/HTTPS URL pointing to a medical image

        Returns:
            tuple: (AnalysisResponse, original_image, ai_attention_heatmap)
            - AnalysisResponse: Complete analysis results
            - original_image: Original PIL Image
            - ai_attention_heatmap: AI Attention Heatmap (colored overlay)
            
        Raises:
            HTTPException: If URL fetch or analysis fails
        """
        try:
            # Fetch image with security validation
            image, image_metadata = await self.fetch_image_from_url(image_url)
            
            # Perform AI analysis
            raw_response, findings, ai_attention_heatmap = await self.analyze_image(image)
            
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
            
            return analysis_response, image, ai_attention_heatmap
            
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
# initialize_database()


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
        pdf_generator.generate_pdf_report,
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

# The working path for the PDF report generation
@app.post("/analyze-url-and-report", response_model=PDFGenerationResponse)
async def analyze_url_and_generate_report(
    request: ImageURLReportRequest
):
    """
    Analyze medical image from URL using MULTI-MODEL analysis and generate PDF report.
    
    Uses both MedGemma-4B-IT and NV-Reason-CXR-3B for consensus-based findings.
    Heatmaps are generated from NV-Reason-CXR attention weights (more reliable).
    
    Args:
        request: ImageURLReportRequest containing the image URL and UUID
        
    Returns:
        PDFGenerationResponse: PDF file S3 path and analysis identifier
    """
    # Perform MULTI-MODEL analysis - uses both MedGemma and NV-Reason-CXR
    multi_response, original_image, heatmap, saliency, probability, segmentation = \
        await multi_analyzer.analyze_image_from_url(str(request.image_url))
    
    # Create adapter response for PDF generator (expects AnalysisResponse format)
    # Map consensus findings to RadiologyFindings format
    consensus = multi_response.consensus
    adapter_findings = RadiologyFindings(
        anatomical_structures=["Lungs", "Heart", "Mediastinum", "Pleura"],
        abnormalities=consensus.all_abnormalities if consensus.all_abnormalities else ["No significant abnormalities detected"],
        impression=consensus.clinical_impression,
        recommendations=[
            "Clinical correlation recommended",
            "Specialist review if symptoms persist"
        ] + (["NV-Reason reasoning available in full report"] if consensus.nv_reason_thinking else []),
        confidence_level=f"{consensus.consensus_level.upper()} (Multi-Model)"
    )
    
    # Create adapter response object for PDF generator
    class PDFAdapterResponse:
        def __init__(self, analysis_id, timestamp, findings, raw_response, image_metadata):
            self.analysis_id = analysis_id
            self.timestamp = timestamp
            self.findings = findings
            self.raw_response = raw_response
            self.image_metadata = image_metadata
    
    # Build combined raw response including NV-Reason thinking chain
    combined_raw = f"""=== MULTI-MODEL ANALYSIS ===

--- MedGemma-4B-IT Analysis ---
{consensus.medgemma_raw if consensus.medgemma_raw else 'Not available'}

--- NV-Reason-CXR-3B Reasoning ---
<thinking>
{consensus.nv_reason_thinking if consensus.nv_reason_thinking else 'Not available'}
</thinking>

<answer>
{consensus.nv_reason_answer if consensus.nv_reason_answer else 'Not available'}
</answer>

--- CONSENSUS ({consensus.consensus_level.upper()}) ---
{consensus.clinical_impression}

Abnormalities agreed by both models: {', '.join(consensus.consensus_abnormalities) if consensus.consensus_abnormalities else 'None'}
All detected abnormalities: {', '.join(consensus.all_abnormalities) if consensus.all_abnormalities else 'None'}
Support devices detected: {', '.join(consensus.nv_reason_devices) if consensus.nv_reason_devices else 'None'}
"""
    
    adapter_response = PDFAdapterResponse(
        analysis_id=multi_response.analysis_id,
        timestamp=multi_response.timestamp,
        findings=adapter_findings,
        raw_response=combined_raw,
        image_metadata=multi_response.image_metadata
    )
    
    # Generate PDF with multi-model heatmap (from NV-Reason-CXR)
    local_pdf_path = await run_in_threadpool(
        pdf_generator.generate_pdf_report, 
        adapter_response,
        original_image,
        heatmap  # This is now from NV-Reason-CXR (primary) or fallback
    )
    
    # LOCAL TESTING MODE: Skip S3 upload and database storage
    # PDF is saved locally in reports/ directory
    logger.info(f"PDF report generated locally: {local_pdf_path}")
    
    # TODO: Re-enable for production
    # s3_bucket_name = os.getenv("AWS_S3_BUCKET_NAME", "your-s3-bucket-name")
    # s3_region = os.getenv("AWS_REGION", "your-aws-region")
    # s3_key = f"reports/{multi_response.analysis_id}/{os.path.basename(local_pdf_path)}"
    # s3_pdf_url = await upload_to_s3(...)
    # await store_analysis_result(image_uuid=request.image_uuid, reportPath=s3_key)
    
    return PDFGenerationResponse(
        pdf_path=local_pdf_path,  # Return local path for testing
        analysis_id=multi_response.analysis_id
    )


@app.get("/model-status")
async def get_model_status():
    """
    Get status of BOTH models (MedGemma and NV-Reason-CXR).
    
    Returns detailed information about model states, device configuration,
    memory usage, and any loading errors for monitoring and debugging.
    """
    global model_loading_error
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    memory_info = analyzer.get_system_memory_info() if analyzer else "unavailable"
    
    # Get status of both models
    multi_status = multi_analyzer.get_model_status() if multi_analyzer else {
        "medgemma": {"loaded": False},
        "nv_reason": {"loaded": False}
    }
    
    return {
        "mode": "multi-model",
        "device": device_str,
        "memory_usage": memory_info,
        "models": {
            "medgemma": {
                "loaded": analyzer.model_loaded if analyzer else False,
                "model_id": analyzer.model_id if analyzer else "not initialized"
            },
            "nv_reason_cxr": multi_status.get("nv_reason", {"loaded": False}),
        },
        "all_models_loaded": (
            (analyzer.model_loaded if analyzer else False) and 
            multi_status.get("nv_reason", {}).get("loaded", False)
        ),
        "error": model_loading_error
    }


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
