"""
Medical Image Analyzer - Core analysis engine.

Main class for AI-powered medical image analysis using MedGemma.
"""

import os
import gc
import io
import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import torch
from PIL import Image
from fastapi import UploadFile, HTTPException
from transformers import AutoProcessor, AutoModelForImageTextToText

from src.config.settings import settings
from src.api.schemas import RadiologyFindings, AnalysisResponse
from src.core.constants import MEDGEMMA_IMAGE_SIZE
from src.services.response_parser import ResponseParser
from src.services.image_fetcher import ImageFetcher
from src.visualization.heatmap import HeatmapGenerator
from src.visualization.saliency import SaliencyGenerator
from src.visualization.localization import LocalizationGenerator
from src.utils.memory import get_system_memory_info

logger = logging.getLogger(__name__)

# Global state for model-loading error
model_loading_error: Optional[str] = None


class MedicalImageAnalyzer:
    """
    Medical image analysis system using Med-GEMMA-3 for AI-powered radiology insights.
    
    This class handles the complete pipeline for medical image analysis including:
    - Model loading and management with CPU/GPU optimization
    - Image preprocessing and validation
    - AI-powered analysis using Med-GEMMA-3
    - Memory management for optimal performance
    """
    
    def __init__(self):
        """Initialize the medical image analyzer."""
        self.model_id = settings.model_id
        self.processor: Optional[Any] = None
        self.model: Optional[Any] = None
        self.reports_dir = settings.reports_dir
        self.max_image_size = settings.max_file_size
        self.allowed_extensions = set(settings.allowed_extensions)
        self.model_loaded = False
        self._last_raw_heatmap: Optional[Any] = None
        
        # Initialize components
        self.response_parser = ResponseParser()
        self.image_fetcher = ImageFetcher(max_image_size=self.max_image_size)
        self.heatmap_generator: Optional[HeatmapGenerator] = None
        self.saliency_generator = SaliencyGenerator()
        self.localization_generator = LocalizationGenerator()
        
        self.ensure_reports_directory()
    
    def ensure_reports_directory(self) -> None:
        """Ensure the reports directory exists for PDF storage."""
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)
    
    async def load_model(self) -> None:
        """
        Load the Med-GEMMA-3 model and processor with memory optimization.
        """
        global model_loading_error
        if self.model_loaded:
            logger.info("Model already loaded.")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model on device: {device}")
        logger.info(f"System memory before loading: {get_system_memory_info()}")

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                device_map="auto"
            )
            self.model_loaded = True
            model_loading_error = None
            
            # Initialize heatmap generator with processor
            self.heatmap_generator = HeatmapGenerator(processor=self.processor)
            
            logger.info("Model loaded successfully.")
            logger.info(f"System memory after loading: {get_system_memory_info()}")
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
    
    async def analyze_image(
        self, 
        image: Image.Image
    ) -> Tuple[str, RadiologyFindings, Image.Image]:
        """
        Perform AI-powered medical image analysis using Med-GEMMA.
        
        Returns:
            Tuple of (raw_response, findings, heatmap_image)
        """
        await self.load_model()

        try:
            # Resize image to MedGemma's expected resolution
            image_for_model = image.resize((MEDGEMMA_IMAGE_SIZE, MEDGEMMA_IMAGE_SIZE))

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
                except Exception as e:
                    logger.warning(f"Forward pass failed: {e}")

                generation_output = self.model.generate(
                    **inputs,
                    max_new_tokens=settings.max_tokens,
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
                                logger.info(f"✅ Got attentions from generation.{attr_name}")
                                break

            generation = generation_output.sequences[0][input_len:]
            raw_response = self.processor.decode(generation, skip_special_tokens=True)

            # Generate heatmap
            heatmap_image, raw_heatmap_np = self.heatmap_generator.generate_heatmap_and_raw_normalized(
                attentions, image
            )
            self._last_raw_heatmap = raw_heatmap_np

            # Parse response
            findings = self.response_parser.parse_ai_response_multi(raw_response)
            return raw_response, findings, heatmap_image

        except Exception as e:
            logger.error(f"Error during image analysis: {str(e)}", exc_info=True)
            placeholder_img = self.heatmap_generator.create_placeholder_heatmap(image)
            self._last_raw_heatmap = None
            findings = RadiologyFindings(
                anatomical_structures=["Error during analysis"],
                abnormalities=[f"Error: {str(e)}"],
                impression="Analysis Incomplete",
                recommendations=["Review error logs"],
                confidence_level="N/A"
            )
            return f"Error: {str(e)}", findings, placeholder_img

    def extract_condition_confidences(self, raw_response: str) -> Dict[str, float]:
        """Extract confidence scores for each medical condition from model text."""
        return self.response_parser.extract_condition_confidences(raw_response)

    async def analyze_image_from_upload(
        self, 
        file: UploadFile
    ) -> Tuple[AnalysisResponse, Image.Image, Image.Image]:
        """
        Analyze medical image from uploaded file and return analysis response with heatmap.
        """
        analysis_id = str(uuid.uuid4())
        
        try:
            contents = await file.read()
            if len(contents) > self.max_image_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large. Maximum size: {self.max_image_size/1024/1024:.1f}MB"
                )
            
            try:
                image = Image.open(io.BytesIO(contents)).convert("RGB")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
            
            image_metadata = {
                "source": "upload",
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(contents),
                "dimensions": image.size,
            }
            
            raw_response, findings, heatmap = await self.analyze_image(image)
            
            response = AnalysisResponse(
                analysis_id=analysis_id,
                timestamp=datetime.now().isoformat(),
                findings=findings,
                raw_response=raw_response,
                image_metadata=image_metadata
            )
            
            return response, image, heatmap
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error analyzing uploaded image: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def analyze_image_from_url(
        self, 
        image_url: str
    ) -> Tuple[AnalysisResponse, Image.Image, Image.Image]:
        """
        Analyze medical image from URL and return analysis response with heatmap.
        """
        analysis_id = str(uuid.uuid4())
        
        image, image_metadata = await self.image_fetcher.fetch_image_from_url(image_url)
        raw_response, findings, heatmap = await self.analyze_image(image)
        
        response = AnalysisResponse(
            analysis_id=analysis_id,
            timestamp=datetime.now().isoformat(),
            findings=findings,
            raw_response=raw_response,
            image_metadata=image_metadata
        )
        
        return response, image, heatmap

    async def fetch_image_from_url(
        self, 
        image_url: str
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Securely fetch medical image from HTTP/HTTPS URL."""
        return await self.image_fetcher.fetch_image_from_url(image_url)

    def get_last_raw_heatmap(self) -> Optional[Any]:
        """Get the last generated raw heatmap for visualization purposes."""
        return self._last_raw_heatmap

    # Visualization helper methods (delegate to generators)
    def generate_saliency_map(self, heatmap_normalized_np, original_image):
        """Generate a saliency map."""
        return self.saliency_generator.generate_saliency_map(heatmap_normalized_np, original_image)

    def generate_density_saliency_map(self, original_image, gaussian_kernel=25, overlay_alpha=0.4):
        """Generate a density saliency map."""
        return self.saliency_generator.generate_density_saliency_map(
            original_image, gaussian_kernel, overlay_alpha
        )

    def generate_probability_map(self, heatmap_normalized_np, original_image):
        """Generate a probability map."""
        return self.saliency_generator.generate_probability_map(heatmap_normalized_np, original_image)

    def generate_localization_map(self, heatmap_raw, original_image, **kwargs):
        """Generate a localization map."""
        return self.localization_generator.generate_localization_map(
            heatmap_raw, original_image, **kwargs
        )

    def generate_segmentation_map(self, heatmap_normalized_np, original_image):
        """Generate a segmentation map."""
        return self.localization_generator.generate_segmentation_map(
            heatmap_normalized_np, original_image
        )
