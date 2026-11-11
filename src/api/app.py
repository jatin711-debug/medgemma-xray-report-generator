import os
import logging
from datetime import datetime
from contextlib import asynccontextmanager

os.environ['TORCH_COMPILE_DISABLE'] = '1'
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from src.api.schemas import (ImageURLRequest, ImageURLReportRequest, RadiologyFindings,
                             AnalysisResponse, PDFGenerationResponse, ModelStatus)
from src.core.medical_analyzer import MedicalImageAnalyzer
from src.reports.generator import UltimatePDFGenerator
from db import store_analysis_result
from aws_utils import upload_to_s3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
analyzer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer
    logger.info("Starting experimental MedGemma API")
    analyzer = MedicalImageAnalyzer()
    await analyzer.load_model()
    yield
    logger.info("Shutting down experimental MedGemma API")

app = FastAPI(title="Medical Image Analysis API (experimental beta)",
              description="Research-only MedGemma chest X-ray prototype",
              version="1.0.0", lifespan=lifespan)
pdf_generator = UltimatePDFGenerator("reports")

# Initialize analyzer
analyzer = MedicalImageAnalyzer()
# Database setup is managed separately; local usage does not require MySQL.


@app.get("/")
async def root():
    """
    Health check endpoint for API status verification.
    
    Returns:
        dict: API status and basic information
    """
    return {"message": "Experimental MedGemma API", "status": "ready" if analyzer and analyzer.model_loaded else "model unavailable"}


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
    if not (file.content_type or "").startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Perform analysis once - this includes heatmap generation
    analysis_response, original_image, heatmap_image, saliency_map, probability_map, segmentation_map = await analyzer.analyze_image_from_upload(file)
    
    # Generate PDF in thread pool to avoid blocking the event loop
    pdf_path = await run_in_threadpool(
        pdf_generator.generate_pdf_report,
        analysis_response,
        original_image,
        heatmap_image, saliency_map, probability_map, segmentation_map
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
    Analyze medical image from HTTP/HTTPS URL using MedGemma.
    
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
    analysis_response, *_ = await analyzer.analyze_image_from_url(str(request.image_url))
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
        raise HTTPException(status_code=502, detail="Failed to upload report to S3") from e

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
    Get current MedGemma model loading status and system information.
    
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
    
    try:
        if analyzer.model is not None and analyzer.model_loading_error is None:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            memory_info = analyzer.get_system_memory_info()
            return ModelStatus(
                loaded=True,
                device=device_str,
                memory_usage=memory_info,
                error=None
            )
        elif analyzer.model_loading_error is not None:
            return ModelStatus(
                loaded=False,
                device="unknown",
                memory_usage=analyzer.get_system_memory_info(),
                error=analyzer.model_loading_error
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
    Force reload the MedGemma pipeline for recovery from loading failures.
    
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
    analyzer.model_loading_error = None
    
    # Reset pipeline for clean reload
    analyzer.model = None
    analyzer.processor = None
    analyzer.model_loaded = False
    
    try:
        await analyzer.load_model()
        return {"message": "Model loading initiated", "status": "success"}
    except Exception as e:
        return {"message": f"Model loading failed: {str(e)}", "status": "error"}


