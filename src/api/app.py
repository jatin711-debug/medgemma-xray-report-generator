import os
import logging
from contextlib import asynccontextmanager

os.environ['TORCH_COMPILE_DISABLE'] = '1'
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
from src.api.schemas import (ImageURLRequest, ImageURLReportRequest, RadiologyFindings,
                             AnalysisResponse, PDFGenerationResponse, ModelStatus)
from src.core.medical_analyzer import MedicalImageAnalyzer
from src.api.security import configured_keys, reader_access, operator_access, admin_access
from src.api.limits import RequestSizeLimitMiddleware, inference_slot
from src.reports.generator import UltimatePDFGenerator
from src.reports.storage import list_local_reports
from db import store_analysis_result
from aws_utils import upload_to_s3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
analyzer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer
    configured_keys()  # Fail before loading the model if access control is invalid.
    logger.info("Starting experimental MedGemma API")
    analyzer = MedicalImageAnalyzer()
    await analyzer.load_model()
    yield
    logger.info("Shutting down experimental MedGemma API")

app = FastAPI(title="Medical Image Analysis API (experimental beta)",
              description="Research-only MedGemma chest X-ray prototype",
              version="1.0.0", lifespan=lifespan,
              docs_url="/docs" if os.getenv("ENABLE_API_DOCS", "").lower() == "true" else None,
              redoc_url=None,
              openapi_url="/openapi.json" if os.getenv("ENABLE_API_DOCS", "").lower() == "true" else None)
app.add_middleware(RequestSizeLimitMiddleware)
pdf_generator = UltimatePDFGenerator("reports")

# Initialize analyzer
analyzer = MedicalImageAnalyzer()
# Database setup is managed separately; local usage does not require MySQL.


@app.get("/health/live", include_in_schema=False)
async def liveness():
    """Unauthenticated process liveness probe; does not report model readiness."""
    return {"status": "alive"}


@app.get("/health/ready", dependencies=[reader_access])
async def readiness():
    """Report model readiness; storage integrations are not checked."""
    if not analyzer or not analyzer.model_loaded:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return {"status": "ready"}


@app.get("/", dependencies=[reader_access])
async def root():
    """
    Health check endpoint for API status verification.
    
    Returns:
        dict: API status and basic information
    """
    return {"message": "Experimental MedGemma API", "status": "ready" if analyzer and analyzer.model_loaded else "model unavailable"}


@app.post("/analyze-and-report", response_model=PDFGenerationResponse, dependencies=[operator_access, Depends(inference_slot)])
async def analyze_and_generate_report(
    file: UploadFile = File(...)
):
    """
    Analyze medical image and generate comprehensive PDF report.
    
    Runs one research analysis and writes its PDF report locally.
    
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
    
    # Perform analysis once
    analysis_response, original_image, attribution = await analyzer.analyze_image_from_upload(file)
    
    # Generate PDF in thread pool to avoid blocking the event loop
    pdf_path = await run_in_threadpool(
        pdf_generator.generate_pdf_report,
        analysis_response,
        original_image,
        attribution
    )
    
    return PDFGenerationResponse(
        pdf_path=pdf_path,
        analysis_id=analysis_response.analysis_id
    )

@app.get("/reports", dependencies=[reader_access])
async def list_reports():
    """
    List all generated medical reports with metadata.
    
    Returns:
        dict: List of reports with filenames, IDs, timestamps, and sizes
        
    Raises:
        HTTPException: If listing operation fails
    """
    try:
        reports = await run_in_threadpool(list_local_reports, pdf_generator.reports_dir)
        return {"reports": reports}
    except Exception as e:
        logger.error(f"Error listing reports: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list reports")


@app.post("/analyze-url", response_model=AnalysisResponse, dependencies=[operator_access, Depends(inference_slot)])
async def analyze_image_url(request: ImageURLRequest):
    """
    Analyze medical image from HTTP/HTTPS URL using MedGemma.
    
    Analyze an image from an enabled public URL.
    
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
@app.post("/analyze-url-and-report", response_model=PDFGenerationResponse, dependencies=[operator_access, Depends(inference_slot)])
async def analyze_url_and_generate_report(
    request: ImageURLReportRequest
):
    """
    Analyze medical image from URL and generate comprehensive PDF report.
    
    Combines image downloading, AI analysis, PDF generation, S3 upload,
    and database logging for the optional integration.
    
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
    s3_bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
    s3_region = os.getenv("AWS_REGION")
    if not s3_bucket_name or not s3_region:
        raise HTTPException(status_code=503, detail="S3 storage is not configured")

    # Perform analysis once
    analysis_response, original_image, attribution = await analyzer.analyze_image_from_url(str(request.image_url))
    
    # Generate PDF in thread pool to avoid blocking the event loop
    local_pdf_path = await run_in_threadpool(
        pdf_generator.generate_pdf_report,
        analysis_response,
        original_image,
        attribution
    )
    
    # Create a unique S3 key, e.g., using the analysis ID
    s3_key = f"reports/{analysis_response.analysis_id}/{os.path.basename(local_pdf_path)}"
    
    try:
        # Upload the PDF to S3
        await upload_to_s3(
            local_file_path=local_pdf_path,
            bucket_name=s3_bucket_name,
            s3_key=s3_key,
            region_name=s3_region
        )
        logger.info("Successfully uploaded PDF to S3")
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


@app.get("/model-status", response_model=ModelStatus, dependencies=[reader_access])
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
                error="Model loading failed; check server logs"
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
            error="Model status unavailable"
        )


@app.post("/load-model", dependencies=[admin_access, Depends(inference_slot)])
async def force_load_model():
    """
    Force reload the MedGemma pipeline for recovery from loading failures.
    
    Useful for recovering from initial loading failures or reinitializing
    the model after configuration changes.
    
    Returns:
        dict: Loading status and result message
        
    Reload waits for any in-progress inference to finish.
    """
    try:
        await analyzer.reload_model()
        return {"message": "Model loaded", "status": "success"}
    except Exception:
        logger.exception("Model reload failed")
        raise HTTPException(status_code=503, detail="Model reload failed")
