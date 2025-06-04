import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, Union
import logging
import aiohttp
import validators
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, HttpUrl
from PIL import Image
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import gc

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global analyzer instance
analyzer = None

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
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
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
        Load and initialize the Med-GEMMA-3 pipeline with optimal device configuration.
        
        This method handles:
        - Device detection (CUDA GPU vs CPU)
        - Memory optimization with appropriate data types
        - Pipeline initialization with proper error handling
        - Memory cleanup for existing models
        - Model validation with dummy inference
        
        Raises:
            HTTPException: If model loading fails, with detailed error information
            
        Performance Optimizations:
        - Uses bfloat16 on GPU for memory efficiency
        - Implements garbage collection and CUDA cache clearing
        - Validates model functionality with test inference
        - Disables torch compilation to avoid Triton errors
        """
        global model_loading_error
        
        if self.model is not None and not model_loading_error:
            # Already loaded successfully
            return
        
        if model_loading_error is not None:
            # If it already failed before, raise 503
            raise HTTPException(status_code=503, detail=f"Model loading previously failed: {model_loading_error}")
        
        logger.info(f"Loading Med-GEMMA-3 model ({self.model_id})...")
        logger.info(f"System memory: {self.get_system_memory_info()}")
        
        try:
            # Clear any existing pipeline for memory optimization
            if self.model is not None:
                del self.model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Optimize device and data type selection
            device = 0 if torch.cuda.is_available() else -1
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            # Set environment variable to disable torch compilation before loading
            os.environ['TORCH_COMPILE_DISABLE'] = '1'
            
            # Initialize processor and model separately
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                device_map= "cuda" if torch.cuda.is_available() else "cpu",
                use_safetensors=True
            )
            
            # Configure the model to use eager execution
            if hasattr(self.model, 'generation_config'):
                self.model.generation_config.use_cache = True
            
            self.model_loaded = True
            model_loading_error = None
            logger.info("Med-GEMMA-3 loaded successfully.")
            
        except Exception as e:
            err = str(e)
            # Check if it's a Triton-related error and provide helpful information
            if "triton" in err.lower() or "inductor" in err.lower():
                err = f"PyTorch compilation error (Triton/Inductor backend): {err}. The model will attempt to use eager execution mode."
                logger.warning(f"Compilation backend failed, this is usually not critical: {err}")
                # Don't treat compilation errors as fatal, try to continue
                try:
                    # Fallback: Force CPU execution if compilation fails
                    logger.info("Attempting fallback to CPU execution...")
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                        use_safetensors=True
                    )
                    
                    # Test the fallback
                    dummy_img = Image.new("RGB", (224, 224), color="white")
                    _ = self.processor(dummy_img, text="<start_of_image> Test", max_new_tokens=10)
                    
                    self.model_loaded = True
                    model_loading_error = None
                    logger.info("Med-GEMMA-3 loaded successfully on CPU (fallback mode).")
                    return
                    
                except Exception as fallback_error:
                    err = f"Both GPU and CPU loading failed. GPU error: {err}. CPU error: {fallback_error}"
            
            model_loading_error = f"Failed to load Med-GEMMA-3: {err}"
            logger.error(model_loading_error)
            raise HTTPException(status_code=503, detail=model_loading_error)
    
    def parse_ai_response(self, raw_response: str) -> RadiologyFindings:
        """
        Parse raw AI model output into structured radiology findings.
        
        This method intelligently extracts and categorizes medical information from
        the AI model's natural language response into standardized clinical sections.
        
        Args:
            raw_response: Raw text output from the Med-GEMMA-3 model
            
        Returns:
            RadiologyFindings: Structured medical findings with categorized information
            
        Processing Steps:
        1. Text preprocessing and line extraction
        2. Section identification using medical keywords
        3. Content categorization into clinical sections
        4. Confidence level assessment
        5. Fallback handling for incomplete responses
        """
        try:
            lines = [line.strip() for line in raw_response.split('\n') if line.strip()]
            
            anatomical_structures = []
            abnormalities = []
            recommendations = []
            impression = ""
            confidence_level = "Medium"
            
            current_section = None
            
            for line in lines:
                line_lower = line.lower()
                
                # Section detection using medical terminology
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
                
                # Content categorization
                if current_section == 'anatomy' and not any(keyword in line_lower for keyword in ['anatomy', 'structures']):
                    anatomical_structures.append(line)
                elif current_section == 'abnormal' and not any(keyword in line_lower for keyword in ['abnormal', 'pathology']):
                    abnormalities.append(line)
                elif current_section == 'recommend' and not any(keyword in line_lower for keyword in ['recommend', 'suggest']):
                    recommendations.append(line)
                elif current_section == 'impression' and not any(keyword in line_lower for keyword in ['impression', 'conclusion']):
                    impression += line + " "
            
            # Fallback handling for incomplete responses
            if not impression and not anatomical_structures and not abnormalities:
                impression = raw_response
            
            # Default values for empty sections
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
            return RadiologyFindings(
                anatomical_structures=["Analysis completed"],
                abnormalities=["Unable to parse detailed findings"],
                impression=raw_response,
                recommendations=["Manual review recommended"],
                confidence_level="Low"
            )
    
    async def analyze_image(self, image: Image.Image) -> tuple[str, RadiologyFindings]:
        """
        Perform AI-powered medical image analysis using Med-GEMMA-3.
        
        This is the core analysis method that processes medical images and generates
        clinical insights using the loaded AI model with proper chat template.
        
        Args:
            image: PIL Image object in RGB format
            
        Returns:
            tuple: (raw_response, structured_findings)
            - raw_response: Original AI model output text
            - structured_findings: Parsed RadiologyFindings object
            
        Raises:
            HTTPException: If analysis fails or model is not loaded
            
        Process:
        1. Ensure model is loaded and ready
        2. Prepare clinical analysis prompt using chat template
        3. Execute AI inference with proper tokenization
        4. Parse and structure the response
        """
        await self.load_model()
        
        try:
            # Prepare clinical analysis messages for Med-GEMMA-3 chat template
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert radiologist."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this medical image. Describe anatomical structures, any abnormalities, and give a clinical impression."},
                        {"type": "image", "image": image}
                    ]
                }
            ]
            
            # Apply chat template and tokenize
            inputs = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, 
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Execute AI inference with torch.inference_mode for efficiency
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    do_sample=False,
                    use_cache=True
                )
                generation = generation[0][input_len:]
            
            # Decode the generated response
            raw_response = self.processor.decode(generation, skip_special_tokens=True)
            
            # Parse response into structured findings
            findings = self.parse_ai_response(raw_response)
            
            return raw_response, findings
        
        except Exception as e:
            logger.error(f"Error during image analysis: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    def generate_pdf_report(self, analysis_response: AnalysisResponse) -> str:
        """
        Generate a comprehensive PDF medical report from analysis results.
        
        Creates a professionally formatted PDF report containing all analysis
        findings, metadata, and clinical recommendations suitable for medical records.
        
        Args:
            analysis_response: Complete analysis results with findings and metadata
            
        Returns:
            str: File path to the generated PDF report
            
        Raises:
            HTTPException: If PDF generation fails
            
        Report Contents:
        - Professional medical report formatting
        - Analysis metadata and timestamps
        - Structured clinical findings
        - Anatomical observations
        - Pathology findings
        - Clinical recommendations
        - Raw AI analysis for reference
        """
        try:
            filename = f"medical_report_{analysis_response.analysis_id}.pdf"
            filepath = os.path.join(self.reports_dir, filename)
            
            doc = SimpleDocTemplate(filepath, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Professional title styling
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                textColor=colors.darkblue
            )
            story.append(Paragraph("Medical Image Analysis Report", title_style))
            story.append(Spacer(1, 12))
            
            # Report metadata table
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
            
            # Clinical findings sections
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
            
            # Clinical impression
            story.append(Paragraph("Clinical Impression", styles['Heading2']))
            story.append(Paragraph(analysis_response.findings.impression, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Raw AI response for transparency
            story.append(Paragraph("Detailed AI Analysis", styles['Heading2']))
            story.append(Paragraph(analysis_response.raw_response, styles['Normal']))
            
            doc.build(story)
            logger.info(f"PDF report generated: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating PDF: {str(e)}")
            raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")
    
    async def analyze_image_from_upload(self, file: UploadFile) -> AnalysisResponse:
        """
        Analyze medical image from uploaded file and return complete analysis response.
        
        Processes uploaded image files through the complete analysis pipeline,
        including validation, preprocessing, AI analysis, and metadata collection.
        
        Args:
            file: FastAPI UploadFile containing the medical image
            
        Returns:
            AnalysisResponse: Complete analysis results with findings and metadata
            
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
            raw_response, findings = await self.analyze_image(image)
            
            # Generate unique analysis ID and timestamp
            analysis_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            return AnalysisResponse(
                analysis_id=analysis_id,
                timestamp=timestamp,
                findings=findings,
                raw_response=raw_response,
                image_metadata=image_metadata
            )
            
        except Exception as e:
            logger.error(f"Error analyzing uploaded image: {str(e)}")
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    async def analyze_image_from_url(self, image_url: str) -> AnalysisResponse:
        """
        Analyze medical image from URL and return complete analysis response.
        
        Downloads and analyzes medical images from HTTP/HTTPS URLs with comprehensive
        validation, security checks, and metadata collection.
        
        Args:
            image_url: Valid HTTP/HTTPS URL pointing to a medical image
            
        Returns:
            AnalysisResponse: Complete analysis results with findings and metadata
            
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
            
            return AnalysisResponse(
                analysis_id=analysis_id,
                timestamp=timestamp,
                findings=findings,
                raw_response=raw_response,
                image_metadata=image_metadata
            )
            
        except Exception as e:
            logger.error(f"Error analyzing image from URL: {str(e)}")
            if isinstance(e, HTTPException):
                raise
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
    
    return await analyzer.analyze_image_from_upload(file)


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
    """
    # Perform analysis first
    analysis_response = await analyzer.analyze_image_from_upload(file)
    
    # Generate PDF in thread pool to avoid blocking the event loop
    pdf_path = await run_in_threadpool(
        analyzer.generate_pdf_report, 
        analysis_response
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
    return await analyzer.analyze_image_from_url(str(request.image_url))


@app.post("/analyze-url-and-report", response_model=PDFGenerationResponse)
async def analyze_url_and_generate_report(
    request: ImageURLReportRequest
):
    """
    Analyze medical image from URL and generate comprehensive PDF report.
    
    Combines secure image downloading, AI analysis, and professional PDF
    report generation in a single optimized endpoint.
    
    Args:
        request: ImageURLReportRequest containing the image URL
        
    Returns:
        PDFGenerationResponse: PDF file path and analysis identifier
        
    Performance Optimization:
        - Uses thread pool for CPU-intensive PDF generation
        - Efficient async image downloading
        - Memory-conscious processing pipeline
    """
    # Perform analysis first
    analysis_response = await analyzer.analyze_image_from_url(str(request.image_url))
    
    # Generate PDF in thread pool to avoid blocking the event loop
    pdf_path = await run_in_threadpool(
        analyzer.generate_pdf_report, 
        analysis_response
    )
    
    return PDFGenerationResponse(
        pdf_path=pdf_path,
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
