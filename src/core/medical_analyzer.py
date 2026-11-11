import os
import uuid
import re
import io
import gc
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import aiohttp
import validators
from fastapi import UploadFile, HTTPException
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from src.api.schemas import AnalysisResponse, RadiologyFindings

logger = logging.getLogger(__name__)

class MedicalImageAnalyzer:
    """
    Medical image analysis system using MedGemma for AI-powered radiology insights.
    
    This class handles the complete pipeline for medical image analysis including:
    - Model loading and management with CPU/GPU optimization
    - Image preprocessing and validation
    - AI-powered analysis using MedGemma
    - PDF report generation
    - Memory management for optimal performance
    
    Attributes:
        model_id: The Hugging Face model identifier for MedGemma
        pipeline: The loaded transformers pipeline for image-text-to-text
        reports_dir: Directory for storing generated PDF reports
        max_image_size: Maximum allowed image size in bytes (10MB)
        allowed_extensions: Set of supported image file extensions
        model_loaded: Boolean flag indicating if the model is successfully loaded
    """
    
    def __init__(self):
        # Only one model: MedGemma
        self.model_id = os.getenv("MODEL_ID", "google/medgemma-1.5-4b-it")
        self.processor: Optional[Any] = None
        self.model: Optional[Any] = None
        self.reports_dir = "reports"
        self.max_image_size = 10 * 1024 * 1024  # 10MB
        self.allowed_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp'}
        self.model_loaded = False
        self.model_loading_error: Optional[str] = None
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
        Load the MedGemma model and processor with memory optimization.
        
        This method handles the loading of the MedGemma model and its
        associated processor. It includes several optimizations for memory
        management, including:
        - 4-bit quantization to reduce model size
        - Support for CPU offloading if CUDA is not available
        - Automatic garbage collection to free up unused memory
        
        The method will attempt to load the model on a CUDA-enabled GPU if
        available, falling back to the CPU if necessary.
        """
        if self.model_loaded:
            logger.info("Model already loaded.")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model on device: {device}")
        logger.info(f"System memory before loading: {self.get_system_memory_info()}")

        # Preserve automatic 4-bit CUDA loading by default; allow full precision.
        quantization_mode = os.getenv("MEDGEMMA_QUANTIZE", "auto").lower()
        if quantization_mode not in {"auto", "1", "0"}:
            raise ValueError("MEDGEMMA_QUANTIZE must be auto, 1, or 0")
        use_quantization = torch.cuda.is_available() and quantization_mode != "0"

        try:
            quantization_config = None
            if use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                attn_implementation="eager",
                device_map="auto"  # Automatically handle device placement
            )
            self.model_loaded = True
            self.model_loading_error = None
            logger.info("Model loaded successfully.")
            logger.info(f"System memory after loading: {self.get_system_memory_info()}")
        except Exception as e:
            self.model_loaded = False
            self.model_loading_error = f"Failed to load model: {str(e)}"
            logger.error(self.model_loading_error)
            # Clean up memory on failure
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(self.model_loading_error) from e
    
    def parse_ai_response(self, raw_response: str) -> RadiologyFindings:
        """
        Parse raw AI model output for pneumothorax detection.
        
        This method intelligently extracts the finding and confidence level for pneumothorax.
        
        Args:
            raw_response: Raw text output from the MedGemma model
              Returns:
            RadiologyFindings: Structured medical findings for pneumothorax.
        """
        # Accept only the exact requested finding line; everything else is uncertain.
        match = re.search(r"(?im)^\s*\*\*Finding\*\*\s*:\s*(.*?)\s*$", raw_response)
        value = match.group(1).strip().strip("* ").lower().rstrip(".") if match else ""
        if value == "no pneumothorax detected":
            impression = "No Pneumothorax Detected"
            abnormalities = ["Model output: no pneumothorax detected."]
        elif value == "pneumothorax detected":
            impression = "Pneumothorax Detected"
            abnormalities = ["Model output: pneumothorax detected."]
        else:
            impression = "Indeterminate"
            abnormalities = ["The model response could not be classified reliably."]
        return RadiologyFindings(
            anatomical_structures=["Chest X-ray"], abnormalities=abnormalities,
            impression=impression, recommendations=["Research use only; expert review required."],
            confidence_level="Not available (uncalibrated)",
        )

    async def analyze_image(self, image: Image.Image) -> tuple[str, RadiologyFindings]:
        """
        Perform AI-powered medical image analysis using Med-GEMMA.
        Returns the raw text response and structured findings.
        """
        await self.load_model()

        try:
            # Resize image to match model input expectations (usually 224x224)
            image_for_model = image

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
                                "**Finding**: [Pneumothorax detected | No pneumothorax detected | Indeterminate]\n"
                                "**Brief Reasoning**: [One-sentence justification]"
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
            ).to(self.model.device)

            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
            raw_response = self.processor.decode(output[0][input_len:], skip_special_tokens=True)
            findings = self.parse_ai_response(raw_response)
            return raw_response, findings
        except Exception as e:
            logger.exception("Image analysis failed")
            raise HTTPException(status_code=500, detail="Image analysis failed") from e

    async def analyze_image_from_upload(self, file: UploadFile) -> tuple[AnalysisResponse, Image.Image]:
        """
        Analyze medical image from uploaded file and return complete analysis response with images.
        
        Processes uploaded image files through the complete analysis pipeline,
        including validation, preprocessing, AI analysis, and metadata collection.
        
        Args:
            file: FastAPI UploadFile containing the medical image
            
        Returns:
            tuple: (AnalysisResponse, original_image)
            
        Raises:
            HTTPException: If file processing or analysis fails
            
        Process:
        1. File validation and size checking
        2. Image preprocessing and format conversion
        3. Metadata extraction
        4. AI-powered analysis
        5. Response structuring with unique ID
        """
        try:
            contents = await file.read(self.max_image_size + 1)
            
            # Validate file size
            if len(contents) > self.max_image_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image too large. Maximum size: {self.max_image_size/1024/1024:.1f}MB"
                )
            
            # Process image and convert to RGB
            try:
                source_image = Image.open(io.BytesIO(contents))
                image_format = source_image.format
                image = source_image.convert('RGB')
            except (OSError, ValueError) as e:
                raise HTTPException(status_code=400, detail="Invalid image format") from e
            
            # Extract comprehensive metadata
            image_metadata = {
                "source": "upload",
                "filename": file.filename,
                "size": len(contents),
                "format": image_format,
                "mode": image.mode,
                "dimensions": image.size,
                "content_type": file.content_type
            }
            
            # Perform AI analysis
            raw_response, findings = await self.analyze_image(image)
            
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
            
            return analysis_response, image
            
        except Exception as e:
            logger.error(f"Error analyzing uploaded image: {str(e)}")
            if isinstance(e, HTTPException):            raise
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    async def analyze_image_from_url(self, image_url: str) -> tuple[AnalysisResponse, Image.Image]:
        """
        Analyze medical image from URL and return complete analysis response with images.
        
        Downloads and analyzes medical images from HTTP/HTTPS URLs with
        validation and metadata collection.
        
        Args:
            image_url: Valid HTTP/HTTPS URL pointing to a medical image

        Returns:
            tuple: (AnalysisResponse, original_image)
            
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
            raw_response, findings = await self.analyze_image(image)
            
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
            
            return analysis_response, image
            
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
            if os.getenv("ENABLE_URL_FETCH", "").lower() != "true":
                raise HTTPException(status_code=403, detail="URL fetching is disabled; use image upload")
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

