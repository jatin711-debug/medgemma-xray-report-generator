import os
import asyncio
import uuid
import gc
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from fastapi import UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from src.api.schemas import AnalysisResponse, RadiologyFindings
from src.services.image_input import fetch_image, read_upload
from src.core.findings import parse_ai_response

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
        self._model_lock = asyncio.Lock()
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
        """Load the shared model without blocking the event loop."""
        async with self._model_lock:
            await run_in_threadpool(self._load_model)

    async def reload_model(self) -> None:
        """Reload only after any active inference has completed."""
        async with self._model_lock:
            self.model = None
            self.processor = None
            self.model_loaded = False
            self.model_loading_error = None
            await run_in_threadpool(self._load_model)

    def _load_model(self) -> None:
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
        try:
            if quantization_mode not in {"auto", "1", "0"}:
                raise ValueError("MEDGEMMA_QUANTIZE must be auto, 1, or 0")
            use_quantization = torch.cuda.is_available() and quantization_mode != "0"
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
        """Delegate parsing to the standalone, model-independent parser."""
        return parse_ai_response(raw_response)

    async def analyze_image(self, image: Image.Image) -> tuple[str, RadiologyFindings]:
        """
        Perform AI-powered medical image analysis using Med-GEMMA.
        Returns the raw text response and structured findings.
        """
        async with self._model_lock:
            await run_in_threadpool(self._load_model)
            return await run_in_threadpool(self._analyze_image, image)

    def _analyze_image(self, image: Image.Image) -> tuple[str, RadiologyFindings]:
        """Run synchronous inference while the caller holds the model lock."""
        try:
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

    def _make_response(self, raw_response: str, findings: RadiologyFindings, metadata: dict) -> AnalysisResponse:
        """Build a response using a UTC timestamp and a generated identifier."""
        return AnalysisResponse(
            analysis_id=str(uuid.uuid4()), timestamp=datetime.now(timezone.utc).isoformat(),
            findings=findings, raw_response=raw_response, image_metadata=metadata,
        )

    async def analyze_image_from_upload(self, file: UploadFile) -> tuple[AnalysisResponse, Image.Image]:
        """Validate an uploaded image and run one model analysis."""
        image, metadata = await read_upload(file, self.max_image_size)
        raw_response, findings = await self.analyze_image(image)
        return self._make_response(raw_response, findings, metadata), image

    async def analyze_image_from_url(self, image_url: str) -> tuple[AnalysisResponse, Image.Image]:
        """Fetch an enabled public image URL and run one model analysis."""
        image, metadata = await self.fetch_image_from_url(image_url)
        raw_response, findings = await self.analyze_image(image)
        return self._make_response(raw_response, findings, metadata), image

    async def fetch_image_from_url(self, image_url: str) -> tuple[Image.Image, Dict[str, Any]]:
        """Fetch an image only when the optional URL workflow is enabled."""
        if os.getenv("ENABLE_URL_FETCH", "").lower() != "true":
            raise HTTPException(status_code=403, detail="URL fetching is disabled; use image upload")
        return await fetch_image(image_url, self.max_image_size)
