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
from transformers import AutoProcessor, AutoModelForImageTextToText 

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
        Load and initialize the Med-GEMMA-3 model with optimal device configuration.
        This version is simplified to run in eager mode, avoiding torch.compile errors.
        """
        global model_loading_error
        
        # 1. Check if model is already loaded and healthy
        if self.model_loaded and self.model is not None and model_loading_error is None:
            logger.info("Model is already loaded.")
            return
        
        # 2. Prevent retrying a load that has already failed
        if model_loading_error is not None:
            raise HTTPException(status_code=503, detail=f"Model loading previously failed: {model_loading_error}")
        
        logger.info(f"Loading Med-GEMMA-3 model ({self.model_id})...")
        logger.info(f"System memory: {self.get_system_memory_info()}")
        
        try:
            # 3. Clean up any previous state before loading
            if self.model is not None:
                del self.model
            if self.processor is not None:
                del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 4. Determine optimal device and data type for eager mode
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
            logger.info(f"Attempting to load model on device='{device}' with dtype='{torch_dtype}'")

            # 5. Load the processor and the model with recommended settings
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            # Use "flash_attention_2" for performance if available, but it may not support outputting
            # attention weights. We'll try it first and fall back to a safer implementation.
            attn_implementation = "flash_attention_2" if torch.cuda.is_available() else "sdpa"

            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=torch_dtype,
                    device_map=device,
                    use_safetensors=True,
                    attn_implementation=attn_implementation,
                )
                logger.info(f"Model loaded successfully with '{attn_implementation}' attention implementation.")
            except (ImportError, ValueError) as e:
                logger.warning(
                    f"'{attn_implementation}' attention implementation failed: {e}. "
                    f"Falling back to 'eager' which is more reliable for attention outputs."
                )
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=torch_dtype,
                    device_map=device,
                    use_safetensors=True,
                    attn_implementation="eager",  # Slower but required for attention weights
                )
            
            self.model_loaded = True
            model_loading_error = None
            logger.info(f"Med-GEMMA-3 loaded successfully on '{device}' in eager mode.")
            
        except Exception as e:
            # 6. Catch any exception during loading (OOM, network, etc.)
            err_msg = f"Failed to load Med-GEMMA-3: {e}"
            model_loading_error = err_msg
            # Log the full stack trace for easier debugging
            logger.error(err_msg, exc_info=True) 
            raise HTTPException(status_code=503, detail=model_loading_error)
    
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
    
    async def analyze_image(self, image: Image.Image) -> tuple[str, RadiologyFindings, Image.Image]:
        """
        Perform AI-powered medical image analysis using Med-GEMMA.
        Returns raw text response, structured findings, and a heatmap overlay.
        """
        await self.load_model()

        try:
            # Resize image to match model input expectations (usually 224x224)
            image_for_model = image.resize((224, 224))

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert radiologist."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Provide a comprehensive radiological analysis of this chest image. Note any abnormalities and conclude with a final impression. Specifically assess for pneumothorax and provide a confidence level for its presence or absence."},
                        {"type": "image", "image": image_for_model}
                    ]
                }
            ]

            # Prepare inputs with processor
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
                # Method 1: Try to get attentions from a single forward pass first
                try:
                    # Get attention weights from forward pass (more reliable)
                    forward_outputs = self.model(
                        **inputs,
                        output_attentions=True,
                        use_cache=False  # Disable cache to ensure attention computation
                    )
                    
                    if hasattr(forward_outputs, 'attentions') and forward_outputs.attentions:
                        attentions = forward_outputs.attentions
                        logger.info(f"✅ Got attentions from forward pass: {len(attentions)} layers")
                    else:
                        logger.warning("No attentions from forward pass")
                        
                except Exception as e:
                    logger.warning(f"Forward pass failed: {e}")

                # Generate the response
                generation_output = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    output_attentions=True,
                    return_dict_in_generate=True,
                    use_cache=False  # Ensure clean generation
                )

                # Method 2: Try to get attentions from generation if forward pass failed
                if attentions is None:
                    # Check all possible attention attributes
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
                        # Debug what's available
                        available_attrs = [attr for attr in dir(generation_output) if not attr.startswith('_')]
                        logger.info(f"Available generation output attributes: {available_attrs}")

            # Decode the generated response
            generation = generation_output.sequences[0][input_len:]
            raw_response = self.processor.decode(generation, skip_special_tokens=True)

            # Debug attention information
            if attentions is not None:
                logger.info(f"Attention info - Type: {type(attentions)}, Length: {len(attentions) if hasattr(attentions, '__len__') else 'No len'}")
                if hasattr(attentions, '__len__') and len(attentions) > 0:
                    first_attention = attentions[0] if isinstance(attentions, (list, tuple)) else attentions
                    if hasattr(first_attention, 'shape'):
                        logger.info(f"First attention shape: {first_attention.shape}")
            else:
                logger.warning("⚠️  Still no attentions found - will create placeholder heatmap")

            # Generate heatmap overlay
            heatmap_image = self._generate_heatmap(attentions, image)

            # Parse and return findings
            findings = self.parse_ai_response(raw_response)
            return raw_response, findings, heatmap_image

        except Exception as e:
            logger.error(f"Error during image analysis: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    def _generate_heatmap(self, attentions: Optional[tuple], original_image: Image.Image) -> Image.Image:
        """
        Generates an accurate heatmap from attention weights and overlays it on the original image.
        This version correctly processes attention from multi-layer, multi-head models like MedGemma
        by averaging attention from text tokens to image patches.
        """
        if not attentions or not hasattr(attentions, '__len__') or len(attentions) == 0:
            logger.warning("No attention weights found or in invalid format. Creating placeholder heatmap.")
            return self._create_placeholder_heatmap(original_image)

        try:
            logger.info(f"Initial attention type: {type(attentions)}, length: {len(attentions)}")
            if len(attentions) > 0:
                logger.info(f"Shape of first attention tensor: {attentions[0].shape}, dtype: {attentions[0].dtype}")

            # attentions is a tuple of tensors, one for each layer.
            # Shape of each tensor: (batch, heads, seq_len, seq_len).
            
            # Convert from bfloat16 if needed, as it's not widely supported for CPU operations
            if attentions[0].dtype == torch.bfloat16:
                attentions = tuple(a.float() for a in attentions)
                logger.info(f"Converted attentions to float32. New dtype: {attentions[0].dtype}")

            # Stack all layer attentions: (layers, batch, heads, seq, seq)
            stacked_attentions = torch.stack(attentions)
            logger.info(f"Shape of stacked_attentions: {stacked_attentions.shape}")
            
            # Average over layers and heads to get a single attention map per batch item
            # Resulting shape: (batch, seq, seq)
            avg_attentions = stacked_attentions.mean(dim=(0, 2))
            logger.info(f"Shape of avg_attentions: {avg_attentions.shape}")
            
            # Assume batch size is 1 and move to CPU for numpy conversion
            # Resulting shape: (seq, seq)
            attention_map = avg_attentions.squeeze(0).cpu().detach()
            logger.info(f"Shape of final attention_map: {attention_map.shape}")

            # For ViT-based models, the image is divided into a grid of patches.
            # A common configuration is a 16x16 grid, resulting in 256 patches/tokens.
            num_image_patches = 256
            grid_size = 16
            
            seq_len = attention_map.shape[0]
            num_text_tokens = seq_len - num_image_patches
            logger.info(f"Sequence length: {seq_len}, Num text tokens: {num_text_tokens}, Num image patches: {num_image_patches}")

            if num_text_tokens < 1:
                logger.warning(f"Sequence length ({seq_len}) is not larger than image patches ({num_image_patches}). "
                               f"Cannot determine text tokens to generate heatmap.")
                return self._create_placeholder_heatmap(original_image)

            # We are interested in how much the text tokens (the prompt) attend to the image patch tokens.
            # The image patches are usually appended at the end of the sequence.
            # The text tokens are at the beginning.
            # Slice the attention map to get attentions from text to image patches.
            # Shape: (num_text_tokens, num_image_patches)
            text_to_image_attention = attention_map[:num_text_tokens, num_text_tokens:]
            logger.info(f"Shape of text_to_image_attention: {text_to_image_attention.shape}")
            
            # Average the attention from all text tokens to each image patch.
            # This gives a single score per image patch.
            # Shape: (num_image_patches)
            image_attention = text_to_image_attention.mean(dim=0)
            logger.info(f"Shape of image_attention: {image_attention.shape}")
            
            if image_attention.shape[0] != num_image_patches:
                 logger.warning(f"Resulting image attention vector length ({image_attention.shape[0]}) "
                                f"does not match num_image_patches ({num_image_patches}).")
                 return self._create_placeholder_heatmap(original_image)

            # Reshape the 1D attention vector into a 2D grid that matches the patch layout.
            heatmap = image_attention.reshape(grid_size, grid_size).numpy()

            # Create the visual heatmap overlay and return it.
            return self._create_heatmap_overlay(heatmap, original_image)

        except Exception as e:
            logger.error(f"Error generating heatmap: {e}", exc_info=True)
            return self._create_placeholder_heatmap(original_image)

    def _create_heatmap_overlay(self, heatmap: np.ndarray, original_image: Image.Image) -> Image.Image:
        """
        Overlays a focused and visually appealing heatmap onto the original image.
        """
        logger.info(f"Heatmap stats before processing: min={heatmap.min():.4f}, max={heatmap.max():.4f}, mean={heatmap.mean():.4f}")

        # Threshold to keep only the top 5% of attention scores for a sharper, more focused heatmap
        threshold_value = np.percentile(heatmap, 95)
        heatmap[heatmap < threshold_value] = 0

        # Normalize the thresholded heatmap to the 0-1 range
        if heatmap.max() > heatmap.min():
            heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            heatmap_normalized = np.zeros_like(heatmap)

        heatmap_uint8 = np.uint8(255 * heatmap_normalized)
        
        # Resize heatmap to match the original image size
        heatmap_resized = cv2.resize(heatmap_uint8, original_image.size, interpolation=cv2.INTER_LINEAR)

        # Apply a Gaussian blur to smooth the edges of the heatmap for a more polished look
        heatmap_blurred = cv2.GaussianBlur(heatmap_resized, (25, 25), 0)
        
        # Apply a color map that resembles the desired output (hot -> cold)
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

    async def analyze_image_from_upload(self, file: UploadFile) -> tuple[AnalysisResponse, Image.Image, Image.Image]:
        """
        Analyze medical image from uploaded file and return complete analysis response with images.
        
        Processes uploaded image files through the complete analysis pipeline,
        including validation, preprocessing, AI analysis, and metadata collection.
        
        Args:
            file: FastAPI UploadFile containing the medical image
            
        Returns:
            tuple: (AnalysisResponse, original_image, heatmap_image)
            - AnalysisResponse: Complete analysis results with findings and metadata
            - original_image: Original PIL Image in RGB format
            - heatmap_image: Generated heatmap overlay image
            
        Raises:
            HTTPException: If file processing or analysis fails
            
        Process:
        1. File validation and size checking
        2. Image preprocessing and format conversion
        3. Metadata extraction
        4. AI-powered analysis with heatmap generation
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
            raw_response, findings, heatmap_image = await self.analyze_image(image)
            
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
            
            return analysis_response, image, heatmap_image
            
        except Exception as e:
            logger.error(f"Error analyzing uploaded image: {str(e)}")
            if isinstance(e, HTTPException):            raise
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    async def analyze_image_from_url(self, image_url: str) -> tuple[AnalysisResponse, Image.Image, Image.Image]:
        """
        Analyze medical image from URL and return complete analysis response with images.
        
        Downloads and analyzes medical images from HTTP/HTTPS URLs with comprehensive
        validation, security checks, and metadata collection.
        
        Args:
            image_url: Valid HTTP/HTTPS URL pointing to a medical image

        Returns:
            tuple: (AnalysisResponse, original_image, heatmap_image)
            - AnalysisResponse: Complete analysis results with findings and metadata
            - original_image: Original PIL Image in RGB format
            - heatmap_image: Generated heatmap overlay image
            
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
            raw_response, findings, heatmap_image = await self.analyze_image(image)
            
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
            
            return analysis_response, image, heatmap_image
            
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
    analysis_response, original_image, heatmap_image = await analyzer.analyze_image_from_url(str(request.image_url))
    
    # Generate PDF in thread pool to avoid blocking the event loop
    local_pdf_path = await run_in_threadpool(
        pdf_generator.generate_pdf_report, 
        analysis_response,
        original_image,
        heatmap_image
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
