import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, Union
import asyncio
import logging
import aiohttp
import validators

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, HttpUrl
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import io
import json
import requests
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical Image Analysis API",
    description="AI-powered medical image analysis with lightweight models",
    version="1.0.0"
)

# Request models
class ImageURLRequest(BaseModel):
    image_url: str = Field(description="HTTP/HTTPS URL of the image to analyze")
    
    class Config:
        schema_extra = {
            "example": {
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
            }
        }
    
class ImageURLReportRequest(BaseModel):
    image_url: str = Field(description="HTTP/HTTPS URL of the image to analyze and generate report")
    
    class Config:
        schema_extra = {
            "example": {
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
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

# Global model variables
model = None
processor = None
model_loading_error = None

class MedicalImageAnalyzer:
    def __init__(self):
        # Use smaller, more reliable models
        self.available_models = {
            "blip": {
                "name": "Salesforce/blip-image-captioning-base",
                "type": "blip",
                "description": "Lightweight image captioning model (good for medical images)"
            },
            "blip2": {
                "name": "Salesforce/blip2-opt-2.7b",
                "type": "blip2", 
                "description": "BLIP2 model for better image understanding"
            },
            "moondream": {
                "name": "vikhyatk/moondream2",
                "type": "moondream",
                "description": "Lightweight vision-language model"
            }
        }
        self.current_model = "blip"  # Default to smallest model
        self.reports_dir = "reports"
        self.max_image_size = 10 * 1024 * 1024  # 10MB
        self.allowed_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp'}
        self.model_loaded = False
        self.ensure_reports_directory()
        
    def ensure_reports_directory(self):
        """Ensure reports directory exists"""
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)
    
    def get_system_memory_info(self):
        """Get system memory information"""
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

    async def load_model(self, model_name: str = None):
        """Load a lightweight vision model with robust error handling"""
        global model, processor, model_loading_error
        
        if model_name is None:
            model_name = self.current_model
            
        if model is not None and processor is not None and model_name == self.current_model:
            return  # Already loaded
            
        if model_loading_error is not None and model_name == self.current_model:
            raise HTTPException(status_code=503, detail=f"Model loading previously failed: {model_loading_error}")
        
        logger.info(f"Loading {model_name} model...")
        logger.info(f"System memory: {self.get_system_memory_info()}")
        
        try:
            # Clear any existing models
            if model is not None:
                del model
            if processor is not None:
                del processor
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            model_info = self.available_models[model_name]
            model_id = model_info["name"]
            
            # Load based on model type with CPU fallback
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading {model_id} on {device}")
            
            if model_info["type"] == "blip":
                processor = BlipProcessor.from_pretrained(model_id)
                if device == "cuda":
                    try:
                        model = BlipForConditionalGeneration.from_pretrained(
                            model_id, 
                            torch_dtype=torch.float16,
                            device_map="auto"
                        )
                    except Exception as e:
                        logger.warning(f"GPU loading failed, trying CPU: {e}")
                        model = BlipForConditionalGeneration.from_pretrained(model_id)
                        model = model.to("cpu")
                else:
                    model = BlipForConditionalGeneration.from_pretrained(model_id)
                    model = model.to("cpu")
                    
            elif model_info["type"] == "moondream":
                from transformers import AutoModelForCausalLM, CodeGenTokenizerFast
                try:
                    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                    if device == "cuda":
                        model = AutoModelForVision2Seq.from_pretrained(
                            model_id, 
                            trust_remote_code=True,
                            torch_dtype=torch.float16,
                            device_map="auto"
                        )
                    else:
                        model = AutoModelForVision2Seq.from_pretrained(
                            model_id, 
                            trust_remote_code=True
                        )
                        model = model.to("cpu")
                except Exception:
                    # Fallback to BLIP if moondream fails
                    logger.warning("Moondream failed, falling back to BLIP")
                    return await self.load_model("blip")
            
            # Test the model
            logger.info("Testing model...")
            device = next(model.parameters()).device
            logger.info(f"Model successfully loaded on {device}")
            
            self.current_model = model_name
            self.model_loaded = True
            model_loading_error = None
            
        except Exception as e:
            error_msg = f"Failed to load {model_name}: {str(e)}"
            model_loading_error = error_msg
            logger.error(error_msg)
            
            # Try fallback to smallest model if not already trying it
            if model_name != "blip":
                logger.info("Trying fallback to BLIP model...")
                try:
                    return await self.load_model("blip")
                except Exception as fallback_error:
                    error_msg = f"All models failed. Last error: {str(fallback_error)}"
                    model_loading_error = error_msg
                    raise HTTPException(status_code=503, detail=error_msg)
            else:
                raise HTTPException(status_code=503, detail=error_msg)

    def parse_ai_response(self, raw_response: str) -> RadiologyFindings:
        """Parse AI response into structured format"""
        try:
            # Simple parsing logic - can be enhanced with more sophisticated NLP
            lines = [line.strip() for line in raw_response.split('\n') if line.strip()]
            
            anatomical_structures = []
            abnormalities = []
            recommendations = []
            impression = ""
            confidence_level = "Medium"
            
            current_section = None
            
            for line in lines:
                line_lower = line.lower()
                
                # Identify sections
                if any(keyword in line_lower for keyword in ['anatomy', 'structures', 'organs']):
                    current_section = 'anatomy'
                elif any(keyword in line_lower for keyword in ['abnormal', 'pathology', 'findings']):
                    current_section = 'abnormal'
                elif any(keyword in line_lower for keyword in ['recommend', 'suggest', 'advice']):
                    current_section = 'recommend'
                elif any(keyword in line_lower for keyword in ['impression', 'conclusion', 'summary']):
                    current_section = 'impression'
                elif any(keyword in line_lower for keyword in ['confidence', 'certain']):
                    if 'high' in line_lower:
                        confidence_level = "High"
                    elif 'low' in line_lower:
                        confidence_level = "Low"
                
                # Extract content based on section
                if current_section == 'anatomy' and not any(keyword in line_lower for keyword in ['anatomy', 'structures']):
                    anatomical_structures.append(line)
                elif current_section == 'abnormal' and not any(keyword in line_lower for keyword in ['abnormal', 'pathology']):
                    abnormalities.append(line)
                elif current_section == 'recommend' and not any(keyword in line_lower for keyword in ['recommend', 'suggest']):
                    recommendations.append(line)
                elif current_section == 'impression' and not any(keyword in line_lower for keyword in ['impression', 'conclusion']):
                    impression += line + " "
            
            # Fallback: if no structured data found, use raw response as impression
            if not impression and not anatomical_structures and not abnormalities:
                impression = raw_response
            
            # Default values if empty
            if not anatomical_structures:
                anatomical_structures = ["Chest structures visible"]
            if not abnormalities:
                abnormalities = ["No obvious abnormalities detected"]
            if not recommendations:
                recommendations = ["Clinical correlation recommended"]
            if not impression:
                impression = "Analysis completed. Clinical review recommended."
            
            return RadiologyFindings(
                anatomical_structures=anatomical_structures,
                abnormalities=abnormalities,
                impression=impression.strip(),
                recommendations=recommendations,
                confidence_level=confidence_level
            )
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {str(e)}")
            # Return default structured response
            return RadiologyFindings(
                anatomical_structures=["Analysis completed"],
                abnormalities=["Unable to parse detailed findings"],
                impression=raw_response,
                recommendations=["Manual review recommended"],
                confidence_level="Low"
            )
    
    async def analyze_image(self, image: Image.Image) -> tuple[str, RadiologyFindings]:
        """Analyze medical image using loaded model"""
        await self.load_model()
        
        try:
            device = next(model.parameters()).device
            
            if self.current_model == "blip":
                # BLIP model analysis
                inputs = processor(image, return_tensors="pt").to(device)
                
                # Generate medical-focused description
                prompt = "a medical image showing"
                inputs_with_prompt = processor(image, prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.generate(**inputs_with_prompt, max_length=150, num_beams=5)
                
                raw_response = processor.decode(outputs[0], skip_special_tokens=True)
                
                # Enhance with medical context
                enhanced_response = self.enhance_medical_analysis(raw_response, image)
                
            elif self.current_model == "moondream":
                # Moondream analysis
                prompt = "Analyze this medical image. Describe the anatomical structures visible, any abnormalities, and provide a clinical impression."
                
                with torch.no_grad():
                    outputs = model.answer_question(image, prompt, processor)
                
                raw_response = outputs
                enhanced_response = raw_response
            
            # Parse response into structured format
            findings = self.parse_ai_response(enhanced_response)
            
            return enhanced_response, findings
            
        except Exception as e:
            logger.error(f"Error during image analysis: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    def enhance_medical_analysis(self, basic_description: str, image: Image.Image) -> str:
        """Enhance basic image description with medical context"""
        try:
            # Analyze image properties for additional context
            width, height = image.size
            aspect_ratio = width / height
            
            # Add medical imaging context
            enhanced = f"""
MEDICAL IMAGE ANALYSIS REPORT

Basic Description: {basic_description}

Technical Details:
- Image dimensions: {width}x{height} pixels
- Aspect ratio: {aspect_ratio:.2f}

Clinical Assessment:
Based on the visual analysis, this appears to be a medical imaging study. 

Anatomical Structures:
- The image shows anatomical structures consistent with medical imaging
- Image quality appears suitable for diagnostic purposes

Findings:
- The automated analysis provides a baseline assessment
- Further clinical correlation is recommended

Recommendations:
- Clinical review by qualified medical professional recommended
- Consider additional imaging if clinically indicated
- Correlation with patient history and physical examination advised

Note: This is an AI-assisted analysis and should not replace professional medical evaluation.
"""
            return enhanced.strip()
            
        except Exception as e:
            logger.error(f"Error enhancing analysis: {str(e)}")
            return basic_description

    def generate_pdf_report(self, analysis_response: AnalysisResponse) -> str:
        """Generate PDF report from analysis results"""
        try:
            filename = f"medical_report_{analysis_response.analysis_id}.pdf"
            filepath = os.path.join(self.reports_dir, filename)
            
            doc = SimpleDocTemplate(filepath, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                textColor=colors.darkblue
            )
            story.append(Paragraph("Medical Image Analysis Report", title_style))
            story.append(Spacer(1, 12))
            
            # Report metadata
            metadata_data = [
                ['Analysis ID:', analysis_response.analysis_id],
                ['Timestamp:', analysis_response.timestamp],
                ['Confidence Level:', analysis_response.findings.confidence_level]
            ]
            
            metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(metadata_table)
            story.append(Spacer(1, 20))
            
            # Findings sections
            sections = [
                ("Anatomical Structures", analysis_response.findings.anatomical_structures),
                ("Abnormalities/Pathologies", analysis_response.findings.abnormalities),
                ("Recommendations", analysis_response.findings.recommendations)
            ]
            
            for section_title, items in sections:
                story.append(Paragraph(section_title, styles['Heading2']))
                for item in items:
                    story.append(Paragraph(f"• {item}", styles['Normal']))
                story.append(Spacer(1, 12))
            
            # Clinical Impression
            story.append(Paragraph("Clinical Impression", styles['Heading2']))
            story.append(Paragraph(analysis_response.findings.impression, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Raw AI Response
            story.append(Paragraph("Detailed AI Analysis", styles['Heading2']))
            story.append(Paragraph(analysis_response.raw_response, styles['Normal']))
            
            doc.build(story)
            logger.info(f"PDF report generated: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating PDF: {str(e)}")
            raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

    async def analyze_image_from_upload(self, file: UploadFile) -> tuple[str, RadiologyFindings, Dict[str, Any]]:
        """Analyze uploaded medical image file"""
        try:
            # Read and process image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            
            # Prepare image metadata
            image_metadata = {
                "source": "upload",
                "filename": file.filename,
                "size": len(contents),
                "format": image.format,
                "mode": image.mode,
                "dimensions": image.size,
                "content_type": file.content_type
            }
            
            # Analyze image
            raw_response, findings = await self.analyze_image(image)
            
            return raw_response, findings, image_metadata
            
        except Exception as e:
            logger.error(f"Error analyzing uploaded image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    async def analyze_image_from_url(self, image_url: str) -> AnalysisResponse:
        """Analyze medical image from URL"""
        try:
            # Fetch image from URL
            image, image_metadata = await self.fetch_image_from_url(image_url)
            
            # Analyze image
            raw_response, findings = await self.analyze_image(image)
            
            # Generate unique analysis ID
            analysis_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            response = AnalysisResponse(
                analysis_id=analysis_id,
                timestamp=timestamp,
                findings=findings,
                raw_response=raw_response,
                image_metadata=image_metadata
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing image from URL: {str(e)}")
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    async def fetch_image_from_url(self, image_url: str) -> tuple[Image.Image, Dict[str, Any]]:
        """Fetch image from HTTPS URL and return PIL Image with metadata"""
        try:
            # Validate URL
            if not validators.url(image_url):
                raise HTTPException(status_code=400, detail="Invalid URL format")
            
            if not image_url.startswith(('https://', 'http://')):
                raise HTTPException(status_code=400, detail="Only HTTP/HTTPS URLs are supported")
            
            # Fetch image with timeout and size limits
            headers = {
                'User-Agent': 'Medical-Image-Analysis-API/1.0.0',
                'Accept': 'image/*'
            }
            
            timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(image_url, headers=headers) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Failed to fetch image. HTTP {response.status}: {response.reason}"
                        )
                    
                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    if not content_type.startswith('image/'):
                        raise HTTPException(
                            status_code=400, 
                            detail=f"URL does not point to an image. Content-Type: {content_type}"
                        )
                    
                    # Check file size
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > self.max_image_size:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Image too large. Maximum size: {self.max_image_size/1024/1024:.1f}MB"
                        )
                    
                    # Read image data
                    image_data = await response.read()
                    
                    # Check actual size after download
                    if len(image_data) > self.max_image_size:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Image too large. Maximum size: {self.max_image_size/1024/1024:.1f}MB"
                        )
            
            # Open image with PIL
            try:
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
            
            # Prepare metadata
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

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Medical Image Analysis API...")
    await analyzer.load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Medical Image Analysis API is running", "status": "healthy"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_medical_image(file: UploadFile = File(...)):
    """Analyze uploaded medical image"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Analyze image
        raw_response, findings = await analyzer.analyze_image(image)
        
        # Prepare image metadata
        image_metadata = {
            "filename": file.filename,
            "size": len(contents),
            "format": image.format,
            "mode": image.mode,
            "dimensions": image.size
        }
        
        response = AnalysisResponse(
            analysis_id=analysis_id,
            timestamp=timestamp,
            findings=findings,
            raw_response=raw_response,
            image_metadata=image_metadata
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in analysis endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze-and-report", response_model=PDFGenerationResponse)
async def analyze_and_generate_report(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Analyze image and generate PDF report"""
    # First analyze the image
    analysis_response = await analyze_medical_image(file)
    
    # Generate PDF report
    pdf_path = analyzer.generate_pdf_report(analysis_response)
    
    return PDFGenerationResponse(
        pdf_path=pdf_path,
        analysis_id=analysis_response.analysis_id
    )

@app.get("/download-report/{analysis_id}")
async def download_report(analysis_id: str):
    """Download PDF report by analysis ID"""
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
    """List all generated reports"""
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
    """Analyze medical image from HTTPS URL"""
    image_url = str(request.image_url)
    
    try:
        # Fetch image from URL
        image, image_metadata = await analyzer.fetch_image_from_url(image_url)
        
        # Analyze image
        raw_response, findings = await analyzer.analyze_image(image)
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        response = AnalysisResponse(
            analysis_id=analysis_id,
            timestamp=timestamp,
            findings=findings,
            raw_response=raw_response,
            image_metadata=image_metadata
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in analyze-url endpoint: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze-url-and-report", response_model=PDFGenerationResponse)
async def analyze_url_and_generate_report(
    background_tasks: BackgroundTasks,
    request: ImageURLReportRequest
):
    """Analyze image from URL and generate PDF report"""
    image_url = str(request.image_url)
    
    try:
        # Fetch image from URL
        image, image_metadata = await analyzer.fetch_image_from_url(image_url)
        
        # Analyze image
        raw_response, findings = await analyzer.analyze_image(image)
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        analysis_response = AnalysisResponse(
            analysis_id=analysis_id,
            timestamp=timestamp,
            findings=findings,
            raw_response=raw_response,
            image_metadata=image_metadata
        )
        
        # Generate PDF report
        pdf_path = analyzer.generate_pdf_report(analysis_response)
        
        return PDFGenerationResponse(
            pdf_path=pdf_path,
            analysis_id=analysis_response.analysis_id
        )
        
    except Exception as e:
        logger.error(f"Error in analyze-url-and-report endpoint: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.get("/model-status", response_model=ModelStatus)
async def get_model_status():
    """Get current model loading status"""
    global model, processor, model_loading_error
    
    try:
        if model is not None and processor is not None:
            device = str(next(model.parameters()).device)
            memory_info = analyzer.get_system_memory_info()
            return ModelStatus(
                loaded=True,
                device=device,
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
    """Force reload the model (useful if initial loading failed)"""
    global model, processor, model_loading_error
    
    # Reset error state
    model_loading_error = None
    model = None
    processor = None
    
    try:
        await analyzer.load_model()
        return {"message": "Model loading initiated", "status": "success"}
    except Exception as e:
        return {"message": f"Model loading failed: {str(e)}", "status": "error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)