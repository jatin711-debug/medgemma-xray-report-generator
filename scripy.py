"""
Production-Grade Medical Image Analysis API using Med-GEMMA.

This API provides endpoints to analyze medical images (specifically for pneumothorax)
from a file upload or a URL. It generates a comprehensive PDF report that includes
the AI's findings and an explainability heatmap (Grad-CAM) showing the model's
area of focus for its diagnosis.
"""

import os
import uuid
import gc
import io
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from contextlib import asynccontextmanager

# --- Environment and PyTorch Setup ---
os.environ['TORCH_COMPILE_DISABLE'] = '1'
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

# --- Third-Party Imports ---
import aiohttp
import torch
import validators
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, HttpUrl
from PIL import Image

# Transformers and AI Model Imports
from transformers import AutoProcessor, AutoModelForImageTextToText

# Explainable AI (XAI) and Visualization Imports
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

# PDF Generation Imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER


# --- Placeholder for Database and Cloud Storage Utilities ---
# In a real application, these would contain actual logic to interact with your services.

def initialize_database():
    """Placeholder for initializing database connections."""
    logging.info("Database connection placeholder initialized.")
    pass

async def store_analysis_result(analysis_data: Dict[str, Any]):
    """Placeholder for storing analysis results in a database."""
    logging.info(f"Placeholder: Storing analysis ID {analysis_data.get('analysis_id')} to DB.")
    pass

async def upload_to_s3(file_path: str, object_name: str) -> Optional[str]:
    """Placeholder for uploading a file to an S3 bucket."""
    logging.info(f"Placeholder: Uploading {file_path} to S3 as {object_name}.")
    # In a real implementation, this would return the S3 URL.
    # For now, we return the local path for simplicity.
    return f"s3://your-bucket-name/{object_name}"


# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Global State ---
analyzer = None
model_loading_error: Optional[str] = None


# --- Medical Image Analyzer Class ---

class MedicalImageAnalyzer:
    """
    Handles the complete medical image analysis pipeline including model loading,
    AI-powered analysis with Grad-CAM heatmap generation, and PDF reporting.
    """

    def __init__(self):
        self.model_id = "google/medgemma-4b-it"
        self.processor: Optional[Any] = None
        self.model: Optional[Any] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        
        self.reports_dir = "reports"
        self.max_image_size = 15 * 1024 * 1024  # 15MB
        self.model_loaded = False
        os.makedirs(self.reports_dir, exist_ok=True)

    async def load_model(self):
        """Loads and initializes the Med-GEMMA model with optimal configuration."""
        global model_loading_error
        if self.model_loaded:
            return
        if model_loading_error:
            raise HTTPException(status_code=503, detail=f"Model loading previously failed: {model_loading_error}")

        logger.info(f"Loading Med-GEMMA model ({self.model_id}) onto {self.device}...")
        try:
            # Cleanup before loading
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                use_safetensors=True,
                attn_implementation="sdpa"
            )
            # Set to evaluation mode by default
            self.model.eval()
            
            self.model_loaded = True
            logger.info(f"Med-GEMMA model loaded successfully on '{self.device}'.")
        except Exception as e:
            err_msg = f"Failed to load Med-GEMMA: {e}"
            model_loading_error = err_msg
            logger.error(err_msg, exc_info=True)
            raise HTTPException(status_code=503, detail=model_loading_error)

    def parse_ai_response(self, raw_response: str) -> Dict[str, Any]:
        """Parses the raw text from the AI into a structured dictionary."""
        import re
        confidence_match = re.search(r'(\d+\.?\d*)\s?%', raw_response)
        confidence = f"{confidence_match.group(1)}%" if confidence_match else "N/A"

        if "no pneumothorax detected" in raw_response.lower():
            finding = "No Pneumothorax Detected"
        else:
            finding = "Pneumothorax Detected"
            
        return {"impression": finding, "confidence_level": confidence}

    async def analyze_image(self, image: Image.Image) -> tuple[str, Dict[str, Any], Image.Image]:
        """
        Performs AI analysis, generates findings, and creates a Grad-CAM heatmap.
        Returns: (raw_response, findings_dict, heatmap_image)
        """
        if not self.model_loaded:
            await self.load_model()

        # The target layer for Grad-CAM in Vision Transformers is often the last MLP block.
        # IMPORTANT: Run `print(self.model)` once to verify this layer name for your model version.
        target_layer_name = 'vision_tower.vision_model.encoder.layers[-1].mlp'
        
        try:
            # 1. Prepare inputs for the model
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze this chest X-ray specifically for pneumothorax. State if pneumothorax is detected or not. Provide a confidence percentage."},
                    {"type": "image"}
                ]}
            ]
            inputs = self.processor(text=[msg["content"][0]["text"] for msg in messages], images=image, return_tensors="pt").to(self.device, self.torch_dtype)
            input_len = inputs["input_ids"].shape[-1]

            # 2. Generate the text response
            self.model.eval()
            with torch.inference_mode():
                generation = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
            raw_response = self.processor.decode(generation[0][input_len:], skip_special_tokens=True).strip()
            findings = self.parse_ai_response(raw_response)

            # 3. Generate the Grad-CAM Heatmap
            heatmap_image = None
            self.model.train()  # Switch to train mode for gradient computation
            with torch.enable_grad():
                cam_extractor = GradCAM(self.model, target_layer=target_layer_name)
                # Re-run the model to get the logits required for CAM
                model_output = self.model(**inputs)
                # Target the generated tokens
                class_indices = generation[0][input_len:]
                saliency_map = cam_extractor(class_indices.tolist(), scores=model_output.logits)[0]
            
            self.model.eval() # CRITICAL: Switch back to evaluation mode
            
            # Overlay heatmap on the original image
            original_tensor = self.processor(images=image, return_tensors="pt").pixel_values
            heatmap_image = overlay_mask(to_pil_image(original_tensor[0]), to_pil_image(saliency_map.squeeze(0), mode='F'), alpha=0.5)

            # Cleanup to free VRAM
            cam_extractor.remove_hooks()
            del cam_extractor, saliency_map, model_output
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return raw_response, findings, heatmap_image

        except Exception as e:
            self.model.eval() # Ensure model is back in eval mode on error
            logger.error(f"Error during analysis or heatmap generation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"AI Analysis failed: {e}")

    async def fetch_image_from_url(self, image_url: str) -> tuple[Image.Image, Dict[str, Any]]:
        """Securely fetches an image from a URL and gathers metadata."""
        if not validators.url(image_url) or not image_url.lower().startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid or non-HTTP/HTTPS URL provided.")
        
        headers = {'User-Agent': 'Medical-Analysis-API/1.0'}
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.get(image_url, headers=headers) as response:
                    if response.status != 200:
                        raise HTTPException(status_code=400, detail=f"Failed to fetch image: HTTP {response.status}")
                    
                    content = await response.read()
                    if len(content) > self.max_image_size:
                        raise HTTPException(status_code=400, detail="Image size exceeds limit.")
                    
                    image = Image.open(io.BytesIO(content)).convert('RGB')
                    metadata = {
                        "source": "url", "url": image_url, "size": len(content),
                        "format": image.format, "dimensions": image.size
                    }
                    return image, metadata
            except aiohttp.ClientError as e:
                raise HTTPException(status_code=500, detail=f"Network error fetching image: {e}")

    def generate_pdf_report(self, analysis_id: str, timestamp: str, raw_response: str, 
                            findings: Dict[str, Any], image_metadata: Dict[str, Any],
                            original_image: Image.Image, heatmap_image: Image.Image) -> str:
        """Generates a professional PDF report with findings and comparison images."""
        filepath = os.path.join(self.reports_dir, f"report_{analysis_id}.pdf")
        doc = SimpleDocTemplate(filepath, pagesize=letter, leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch)
        styles = getSampleStyleSheet()
        story = []

        # --- Styles ---
        title_style = ParagraphStyle(name='Title', parent=styles['h1'], fontSize=24, alignment=TA_CENTER, spaceAfter=20, textColor=colors.darkblue)
        subtitle_style = ParagraphStyle(name='Subtitle', parent=styles['h2'], fontSize=16, alignment=TA_CENTER, spaceAfter=10, textColor=colors.gray)
        section_style = ParagraphStyle(name='Section', parent=styles['h2'], fontSize=14, spaceBefore=12, spaceAfter=8, textColor=colors.darkblue)
        body_style = styles['BodyText']
        
        # --- Cover Page ---
        story.append(Paragraph("Pneumothorax Detection Report", title_style))
        story.append(Paragraph("AI-Powered Analysis with Med-GEMMA", subtitle_style))
        story.append(Spacer(1, 0.5 * inch))
        
        summary_data = [
            ['Analysis ID:', analysis_id],
            ['Timestamp:', timestamp],
            ['AI Finding:', f"<b>{findings['impression']}</b>"],
            ['AI Confidence:', f"<b>{findings['confidence_level']}</b>"]
        ]
        summary_table = Table(summary_data, colWidths=[1.5*inch, 5*inch])
        summary_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.darkslategray),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(summary_table)
        story.append(PageBreak())

        # --- Image Visualization Section ---
        story.append(Paragraph("Image Analysis Visualization", section_style))
        story.append(Paragraph("The heatmap highlights regions the AI focused on. Bright areas indicate higher influence on the model's decision.", body_style))
        story.append(Spacer(1, 0.2 * inch))

        def pil_to_rl_image(pil_img, width=2.5*inch):
            img_buffer = io.BytesIO()
            pil_img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            return RLImage(img_buffer, width=width, height=width * (pil_img.height / pil_img.width))

        img_table_data = [
            [Paragraph("<b>Original Image</b>", body_style), Paragraph("<b>AI Focus (Heatmap)</b>", body_style)],
            [pil_to_rl_image(original_image), pil_to_rl_image(heatmap_image)]
        ]
        img_table = Table(img_table_data, colWidths=[3.2*inch, 3.2*inch])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8)
        ]))
        story.append(img_table)
        story.append(PageBreak())

        # --- Details and Disclaimer ---
        story.append(Paragraph("Raw AI Model Output", section_style))
        story.append(Paragraph(raw_response.replace('\n', '<br/>'), styles['Code']))
        
        story.append(Paragraph("Disclaimer", section_style))
        disclaimer = "This AI-generated report is for informational purposes only and is NOT a substitute for professional medical advice from a qualified human expert. Clinical decisions must be made in consultation with a radiologist."
        story.append(Paragraph(disclaimer, ParagraphStyle(name='disclaimer', parent=body_style, textColor=colors.red)))
        
        doc.build(story)
        logger.info(f"PDF report generated: {filepath}")
        return filepath


# --- FastAPI Application Setup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    logger.info("Starting Medical Image Analysis API...")
    global analyzer
    initialize_database()
    analyzer = MedicalImageAnalyzer()
    await analyzer.load_model()
    yield
    logger.info("Shutting down Medical Image Analysis API...")
    # Add any cleanup here
    if os.path.exists(analyzer.reports_dir):
        import shutil
        shutil.rmtree(analyzer.reports_dir)


app = FastAPI(
    title="Medical Image Analysis & Report API",
    description="Analyzes medical images for pneumothorax, generating a PDF report with an AI-focus heatmap.",
    version="2.0.0",
    lifespan=lifespan
)


# --- API Endpoints ---

@app.post(
    "/generate-report/from-upload",
    tags=["Report Generation"],
    summary="Upload an image to generate a full PDF report.",
    response_class=FileResponse
)
async def generate_report_from_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Medical image file (e.g., chest X-ray).")
):
    """
    Accepts an image file upload, performs AI analysis with heatmap generation,
    creates a PDF report, and returns it.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    contents = await file.read()
    if len(contents) > analyzer.max_image_size:
        raise HTTPException(status_code=413, detail="Image file is too large.")
        
    original_image = Image.open(io.BytesIO(contents)).convert('RGB')
    image_metadata = {
        "source": "upload", "filename": file.filename, "size": len(contents),
        "format": original_image.format, "dimensions": original_image.size
    }

    raw_response, findings, heatmap_image = await analyzer.analyze_image(original_image)
    
    analysis_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    pdf_path = analyzer.generate_pdf_report(
        analysis_id, timestamp, raw_response, findings, image_metadata, original_image, heatmap_image
    )
    
    # Store results and upload to S3 in the background
    analysis_data = {"analysis_id": analysis_id, "timestamp": timestamp, "findings": findings, "metadata": image_metadata}
    background_tasks.add_task(store_analysis_result, analysis_data)
    background_tasks.add_task(upload_to_s3, pdf_path, f"reports/{os.path.basename(pdf_path)}")
    
    # Add task to delete local file after sending response
    background_tasks.add_task(os.remove, pdf_path)

    return FileResponse(pdf_path, media_type='application/pdf', filename=os.path.basename(pdf_path))


class URLRequest(BaseModel):
    image_url: HttpUrl = Field(..., description="Publicly accessible URL of the medical image.")

@app.post(
    "/generate-report/from-url",
    tags=["Report Generation"],
    summary="Provide an image URL to generate a full PDF report.",
    response_class=FileResponse
)
async def generate_report_from_url(
    background_tasks: BackgroundTasks,
    request: URLRequest = Body(...)
):
    """
    Accepts an image URL, performs AI analysis with heatmap generation,
    creates a PDF report, and returns it.
    """
    original_image, image_metadata = await analyzer.fetch_image_from_url(str(request.image_url))
    
    raw_response, findings, heatmap_image = await analyzer.analyze_image(original_image)

    analysis_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    pdf_path = analyzer.generate_pdf_report(
        analysis_id, timestamp, raw_response, findings, image_metadata, original_image, heatmap_image
    )

    analysis_data = {"analysis_id": analysis_id, "timestamp": timestamp, "findings": findings, "metadata": image_metadata}
    background_tasks.add_task(store_analysis_result, analysis_data)
    background_tasks.add_task(upload_to_s3, pdf_path, f"reports/{os.path.basename(pdf_path)}")
    background_tasks.add_task(os.remove, pdf_path)
    
    return FileResponse(pdf_path, media_type='application/pdf', filename=os.path.basename(pdf_path))


# --- Main execution block for running the server ---
if __name__ == "__main__":
    import uvicorn
    # To run: uvicorn main:app --reload
    # The --host and --port can be configured as needed.
    uvicorn.run(app, host="0.0.0.0", port=8000)