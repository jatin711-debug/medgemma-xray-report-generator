from builtins import str
import os
import cv2
import numpy as np
import uuid
import re
import io
import gc
from datetime import datetime
from typing import Optional, Dict, Any, Union
from contextlib import asynccontextmanager
import logging
os.environ['TORCH_COMPILE_DISABLE'] = '1'
import aiohttp
import validators
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")  # Load environment variables from .env file

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, HttpUrl
from PIL import Image
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as ReportLabImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
import gc

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, Idefics2ForConditionalGeneration 

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
            
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=torch_dtype,
                    device_map=device,
                    use_safetensors=True,
                    attn_implementation="flash_attention_2"
                )
            except ImportError:
                logger.warning("FlashAttention2 not available. Falling back to 'sdpa'")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=torch_dtype,
                    device_map=device,
                    use_safetensors=True,
                    attn_implementation="sdpa"
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
            image = image.resize((224, 224))

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert radiologist."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this medical image specifically for pneumothorax. "
                                                "Don't give any other information. Just say if pneumothorax is detected or not. "
                                                "If none, say 'No pneumothorax detected'. Give confidence percentage in the response."},
                        {"type": "image", "image": image}
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

            with torch.inference_mode():
                generation_output = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    output_attentions=True,
                    return_dict_in_generate=True
                )

            generation = generation_output.sequences[0][input_len:]
            raw_response = self.processor.decode(generation, skip_special_tokens=True)

            # Try to get decoder attentions (more reliable than full attention)
            attentions = getattr(generation_output, "decoder_attentions", None)

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
        Generates a heatmap from attention weights and overlays it on the original image.
        """
        if not attentions:
            logger.warning("No attention weights found.")
            return original_image

        try:
            # Extract attention tensors (list of layers)
            attention_tensors = [att for att in attentions if isinstance(att, torch.Tensor)]
            if not attention_tensors:
                logger.warning("No valid attention tensors.")
                return original_image

            # Shape: (batch, heads, tgt_len, src_len)
            # We'll average across layers, heads, and target tokens
            stacked = torch.stack(attention_tensors)  # (layers, batch, heads, tgt, src)
            avg_attention = stacked.mean(dim=(0, 1, 2))  # shape: (src_len,) or (src_len,)

            attention_np = avg_attention.cpu().numpy().flatten()
            num_patches = len(attention_np)

            # Try to reshape to square grid
            side = int(np.sqrt(num_patches))
            if side * side != num_patches:
                # Fallback: try best-fit rectangle
                factors = [(i, num_patches // i) for i in range(1, side + 1) if num_patches % i == 0]
                if not factors:
                    logger.warning("Cannot reshape attention to grid. Generating placeholder heatmap.")
                    heatmap = np.ones((64, 64)) * 0.3
                else:
                    h, w = max(factors, key=lambda x: min(x))
                    heatmap = attention_np[:h * w].reshape(h, w)
            else:
                heatmap = attention_np.reshape(side, side)

            # Normalize
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
            heatmap = cv2.resize(heatmap, original_image.size)

            # Colorize
            heatmap_uint8 = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            original_bgr = cv2.cvtColor(np.array(original_image.convert("RGB")), cv2.COLOR_RGB2BGR)
            blended = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)
            blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

            return Image.fromarray(blended_rgb)

        except Exception as e:
            logger.error(f"Heatmap generation failed: {e}", exc_info=True)
            return original_image


    def generate_pdf_report(self, analysis_response: AnalysisResponse, original_image: Image.Image, heatmap_image: Image.Image) -> str:
        """
        Generate a comprehensive, professionally designed PDF medical report.

        This function creates a modern, visually appealing report similar to those
        from major tech companies, with a clear structure, branding, and high readability.
        It now includes a second page for visual comparison of the original image and AI heatmap.

        Args:
            analysis_response: The complete analysis result object.
            original_image: The original PIL image.
            heatmap_image: The generated PIL image with heatmap overlay.

        Returns:
            The file path to the generated PDF report.
        """
        try:
            filename = f"medical_report_{analysis_response.analysis_id}.pdf"
            filepath = os.path.join(self.reports_dir, filename)
            doc = SimpleDocTemplate(filepath, pagesize=letter,
                                    rightMargin=40, leftMargin=40,
                                    topMargin=40, bottomMargin=40)
            
            # <<< FIX: Get the base stylesheet first.
            styles = getSampleStyleSheet()
            story = []

            # --- Define Professional Color Palette ---
            primary_color = colors.HexColor("#174EA6")
            secondary_color = colors.HexColor("#4285F4")
            text_color = colors.HexColor("#3c4043")
            light_bg_color = colors.HexColor("#E8F0FE")
            grey_text = colors.HexColor("#5F6368")
            error_color = colors.HexColor("#EA4335")
            success_color = colors.HexColor("#0B8043")
            link_color = colors.HexColor("#0000EE")

            # <<< FIX: Create all custom ParagraphStyle objects with unique names, then add them.
            # This is the correct and robust way to manage styles.
            styles.add(ParagraphStyle(name='CoverTitle', parent=styles['Title'], fontSize=28,
                                    alignment=TA_CENTER, spaceAfter=20, textColor=primary_color))
            
            styles.add(ParagraphStyle(name='CustomSubtitle', parent=styles['h2'], fontSize=16,
                                    alignment=TA_CENTER, textColor=secondary_color, spaceAfter=10))
            
            styles.add(ParagraphStyle(name='MetaInfo', parent=styles['Normal'], fontSize=10,
                                    alignment=TA_CENTER, textColor=grey_text, spaceAfter=2))
            
            styles.add(ParagraphStyle(name='SectionTitle', parent=styles['h1'], fontSize=16,
                                    textColor=primary_color, spaceAfter=12, keepWithNext=1))
            
            # Create a 'CustomBullet' style to avoid conflict with the default 'Bullet'
            styles.add(ParagraphStyle(name='CustomBullet', parent=styles['Normal'], fontSize=11,
                                    leftIndent=20, bulletIndent=10, spaceAfter=5, textColor=text_color))
            
            styles.add(ParagraphStyle(name='KeyFinding', parent=styles['Normal'], fontSize=12,
                                    textColor=success_color, leftIndent=10, spaceAfter=8, leading=14))
            
            styles.add(ParagraphStyle(name='AIAnalysis', parent=styles['Code'], fontSize=9,
                                    leftIndent=10, textColor=grey_text, spaceAfter=10,
                                    backColor=colors.HexColor("#F1F3F4"), borderPadding=8,
                                    borderColor=colors.HexColor("#DADCE0"), borderWidth=1, borderRadius=4))
            
            styles.add(ParagraphStyle(name='CustomDisclaimer', parent=styles['Normal'], fontSize=9,
                                    textColor=error_color, leftIndent=10, spaceAfter=8, leading=12))
            
            styles.add(ParagraphStyle(name='SummaryBox', parent=styles['Normal'], fontSize=12,
                                    backColor=light_bg_color, borderPadding=12,
                                    borderColor=primary_color, borderWidth=1, borderRadius=6,
                                    spaceAfter=20, textColor=text_color, leading=16))
            
            styles.add(ParagraphStyle(name='LinkStyle', parent=styles['Normal'], fontSize=10, textColor=link_color))


            # --- 1. Single-Page Report ---
            story.append(Spacer(1, 0.5 * inch))
            story.append(Paragraph("Pneumothorax Detection Report", styles['CoverTitle']))
            story.append(Paragraph("AI-Powered Analysis with Med-GEMMA-3", styles['CustomSubtitle']))
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph(f"<b>Analysis ID:</b> {analysis_response.analysis_id}", styles['MetaInfo']))
            story.append(Paragraph(f"<b>Generated:</b> {analysis_response.timestamp}", styles['MetaInfo']))
            story.append(Spacer(1, 0.3 * inch))
            summary_text = (
                f"<b>Finding:</b> {analysis_response.findings.impression}<br/><br/>"
                f"<b>Confidence:</b> {analysis_response.findings.confidence_level}"
            )
            story.append(Paragraph(summary_text, styles['SummaryBox']))
            story.append(Spacer(1, 0.2 * inch))

            # --- Report Details ---
            story.append(Paragraph("Analysis Details", styles['SectionTitle']))
            metadata = analysis_response.image_metadata
            source_type = metadata.get("source")
            source_row = []
            if source_type == 'url':
                image_url = metadata.get('url', 'N/A')
                display_url = (image_url[:70] + '...') if len(image_url) > 70 else image_url
                link_paragraph = Paragraph(f'<link href="{image_url}"><u>{display_url}</u></link>', styles['LinkStyle'])
                source_row = ['Image URL', link_paragraph]
            elif source_type == 'upload':
                source_row = ['Uploaded File', metadata.get('filename', 'N/A')]
            else:
                source_row = ['Image Source', 'Unknown']
            meta_data = [
                ['Analysis ID', analysis_response.analysis_id],
                ['Timestamp', analysis_response.timestamp],
                ['Finding', analysis_response.findings.impression],
                ['Confidence', analysis_response.findings.confidence_level],
                source_row,
                ['Image Format', metadata.get("format", "N/A")],
                ['Image Size', f"{metadata.get('size', 0) / 1024:.1f} KB"],
                ['Dimensions', str(metadata.get("dimensions", "N/A"))],
            ]
            meta_table = Table(meta_data, colWidths=[2.0 * inch, 4.5 * inch])
            meta_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#F1F3F4")),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#DADCE0")),
            ]))
            story.append(meta_table)
            story.append(Spacer(1, 0.2 * inch))

            # --- Detailed AI Analysis (Raw Output) ---
            story.append(Paragraph("Detailed AI Analysis", styles['SectionTitle']))
            story.append(Paragraph("The following is the raw, unformatted text generated by the AI model.", styles['Normal']))
            story.append(Spacer(1, 0.1 * inch))
            raw_text_formatted = analysis_response.raw_response.replace('\n', '<br/>')
            story.append(Paragraph(raw_text_formatted, styles['AIAnalysis']))
            story.append(Spacer(1, 0.2 * inch))

            # --- Disclaimer ---
            story.append(Paragraph("Disclaimer", styles['SectionTitle']))
            disclaimer_text = (
                "This report was generated by an AI model (Med-GEMMA-3) and is intended for informational "
                "purposes only. It is not a substitute for professional medical advice, diagnosis, or "
                "treatment from a qualified human expert. Clinical decisions should be made in "
                "consultation with a radiologist, correlating these findings with the patient's "
                "clinical history and other diagnostic results."
            )
            story.append(Paragraph(disclaimer_text, styles['CustomDisclaimer']))

            # --- PAGE BREAK FOR VISUAL ANALYSIS ---
            story.append(PageBreak())

            # --- 2. Second Page - Visual Comparison ---
            story.append(Paragraph("Visual Analysis: AI Focus Map", styles['SectionTitle']))
            story.append(Paragraph(
                "The following images provide a side-by-side comparison of the original medical scan "
                "and a heatmap visualizing the areas the AI model focused on for its analysis. "
                "The brighter, warmer areas (red, yellow) indicate a higher level of AI attention.",
                styles['Normal']
            ))
            story.append(Spacer(1, 0.2 * inch))

            # Convert PIL images to reportlab Image flowables by saving to a memory buffer
            
            original_img_buffer = io.BytesIO()
            original_image.save(original_img_buffer, format='PNG')
            original_img_buffer.seek(0)
            rl_original_image = ReportLabImage(original_img_buffer, width=3*inch, height=3*inch, kind='proportional')
            
            heatmap_img_buffer = io.BytesIO()
            heatmap_image.save(heatmap_img_buffer, format='PNG')
            heatmap_img_buffer.seek(0)
            rl_heatmap_image = ReportLabImage(heatmap_img_buffer, width=3*inch, height=3*inch, kind='proportional')

            # Create a table for side-by-side comparison
            image_comparison_data = [
                [Paragraph("<b>Original Image</b>", styles['Normal']), Paragraph("<b>AI Heatmap</b>", styles['Normal'])],
                [rl_original_image, rl_heatmap_image]
            ]
            
            image_table = Table(image_comparison_data, colWidths=[3.2*inch, 3.2*inch])
            image_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('TOPPADDING', (0, 1), (-1, 1), 6),
            ]))
            
            story.append(image_table)


            # --- Footer with Page Numbers ---
            def add_footer(canvas, doc):
                canvas.saveState()
                footer_text = "Med-GEMMA-3 | AI Medical Report | Confidential"
                page_num_text = f"Page {doc.page}"
                canvas.setFont('Helvetica', 8)
                canvas.setFillColor(grey_text)
                canvas.drawString(40, 25, footer_text)
                canvas.drawRightString(doc.width + 30, 25, page_num_text)
                canvas.restoreState()

            # Build the PDF (now multi-page)
            doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
            logger.info(f"Professionally styled PDF report generated: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error generating professional PDF: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")
        
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


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_medical_image(file: UploadFile = File(...)):
    """
    Analyze uploaded medical image using Med-GEMMA-3 AI model.
    
    This endpoint processes uploaded medical images and returns structured
    analysis results including anatomical findings, abnormalities, and
    clinical recommendations.
    
    Args:
        file: Uploaded medical image file (JPEG, PNG, TIFF, BMP, WebP)
        
    Returns:
        AnalysisResponse: Complete analysis results with structured findings
        
    Raises:
        HTTPException: For invalid files, processing errors, or analysis failures
        
    Features:
        - File format validation
        - Size limitations (10MB max)
        - AI-powered medical analysis
        - Structured clinical output
        - Comprehensive metadata collection
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    analysis_response, _, _ = await analyzer.analyze_image_from_upload(file)
    return analysis_response


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


@app.get("/download-report/{analysis_id}")
async def download_report(analysis_id: str):
    """
    Download generated PDF medical report by analysis ID.
    
    Args:
        analysis_id: Unique identifier for the analysis and report
        
    Returns:
        FileResponse: PDF file download
        
    Raises:
        HTTPException: If report file is not found
    """
    pdf_filename = f"medical_report_{analysis_id}.pdf"
    pdf_path = os.path.join(analyzer.reports_dir, pdf_filename)
    
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(
        pdf_path,
        media_type='application/pdf',
        filename=pdf_filename
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
        analyzer.generate_pdf_report, 
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
