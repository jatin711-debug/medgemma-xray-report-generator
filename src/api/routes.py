"""
API Routes for Medical Image Analysis.

Contains all endpoint handlers for the Medical Image Analysis API.
"""

import os
import logging
from datetime import datetime

import torch
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool

from src.api.schemas import (
    ImageURLRequest,
    ImageURLReportRequest,
    RadiologyFindings,
    PDFGenerationResponse,
)
from src.api.dependencies import get_analyzer, get_multi_analyzer
from src.reports.clinical_generator import ClinicalReportGenerator
from src.utils.memory import get_system_memory_info

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
async def root():
    """
    Health check endpoint for API status verification.
    
    Returns:
        dict: API status and basic information
    """
    return {
        "message": "Medical Image Analysis API (Med-GEMMA-3) is running", 
        "status": "healthy"
    }


@router.post("/analyze-and-report", response_model=PDFGenerationResponse)
async def analyze_and_generate_report(
    file: UploadFile = File(...)
):
    """
    Analyze medical image and generate comprehensive PDF report.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    multi_analyzer = get_multi_analyzer()
    
    # Perform analysis with Multi-Model Analyzer
    (analysis_response, original_image, heatmap_image, saliency_map, probability_map, 
     segmentation_map, localization_map, detected_regions, condition_confidences, 
     per_condition_maps) = await multi_analyzer.analyze_image_from_upload(file)
    
    # Helper to unpack consensus for report generator
    consensus = analysis_response.consensus
    consensus_data = {
        'medgemma_raw': consensus.medgemma_raw,
        'medgemma_confidence': consensus.medgemma_confidence,
        'clinical_impression': consensus.clinical_impression,
        'all_abnormalities': consensus.all_abnormalities,
        'medgemma_abnormalities': consensus.medgemma_abnormalities,
    }
    
    primary_condition = getattr(analysis_response, 'primary_condition', "Pneumonia")
    
    # Initialize PDF generator
    pdf_generator = ClinicalReportGenerator("reports")
    
    # Generate PDF in thread pool
    pdf_path = await run_in_threadpool(
        pdf_generator.generate_report,
        analysis_response,
        original_image,
        heatmap_image,
        consensus_data,
        saliency_map,
        localization_map,
        detected_regions,
        condition_confidences,
        primary_condition,
        per_condition_maps
    )
    
    return PDFGenerationResponse(
        pdf_path=pdf_path,
        analysis_id=analysis_response.analysis_id
    )


@router.get("/reports")
async def list_reports():
    """
    List all generated medical reports with metadata.
    """
    analyzer = get_analyzer()
    
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


@router.post("/analyze-url-and-report", response_model=PDFGenerationResponse)
async def analyze_url_and_generate_report(
    request: ImageURLReportRequest
):
    """
    Analyze medical image from URL using MULTI-MODEL analysis and generate PDF report.
    """
    multi_analyzer = get_multi_analyzer()
    
    # Perform MULTI-MODEL analysis
    (multi_response, original_image, heatmap, saliency, probability, segmentation,
     localization_map, detected_regions, condition_confidences, per_condition_maps) = \
        await multi_analyzer.analyze_image_from_url(str(request.image_url))
    
    # Create adapter response for PDF generator
    consensus = multi_response.consensus
    adapter_findings = RadiologyFindings(
        anatomical_structures=["Lungs", "Heart", "Mediastinum", "Pleura"],
        abnormalities=consensus.all_abnormalities if consensus.all_abnormalities else ["No significant abnormalities detected"],
        impression=consensus.clinical_impression,
        recommendations=[
            "Clinical correlation recommended",
            "Specialist review if symptoms persist"
        ],
        confidence_level=f"{consensus.medgemma_confidence} (MedGemma)"
    )
    
    # Build consensus data for report
    consensus_data = {
        'medgemma_raw': consensus.medgemma_raw,
        'medgemma_confidence': consensus.medgemma_confidence,
        'clinical_impression': consensus.clinical_impression,
        'all_abnormalities': consensus.all_abnormalities,
        'medgemma_abnormalities': consensus.medgemma_abnormalities,
    }
    
    # Adapter response class for PDF generator
    class PDFAdapterResponse:
        def __init__(self, analysis_id, timestamp, findings, raw_response, image_metadata):
            self.analysis_id = analysis_id
            self.timestamp = timestamp
            self.findings = findings
            self.raw_response = raw_response
            self.image_metadata = image_metadata
    
    # Initialize PDF generator
    pdf_generator = ClinicalReportGenerator()
    
    adapter_response = PDFAdapterResponse(
        analysis_id=multi_response.analysis_id,
        timestamp=multi_response.timestamp,
        findings=adapter_findings,
        raw_response=consensus.medgemma_raw,
        image_metadata=multi_response.image_metadata
    )
    
    # Generate PDF
    primary_condition = getattr(multi_response, 'primary_condition', "Pneumonia")
    local_pdf_path = await run_in_threadpool(
        pdf_generator.generate_report,
        adapter_response,
        original_image,
        heatmap,
        consensus_data,
        saliency,
        localization_map,
        detected_regions,
        condition_confidences,
        primary_condition,
        per_condition_maps
    )
    
    logger.info(f"PDF report generated locally: {local_pdf_path}")
    
    return PDFGenerationResponse(
        pdf_path=local_pdf_path,
        analysis_id=multi_response.analysis_id
    )


@router.get("/model-status")
async def get_model_status():
    """
    Get status of all models (MedGemma and NV-Reason-CXR).
    """
    analyzer = get_analyzer()
    multi_analyzer = get_multi_analyzer()
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    memory_info = get_system_memory_info()
    
    multi_status = multi_analyzer.get_model_status() if multi_analyzer else {
        "medgemma": {"loaded": False},
        "nv_reason": {"loaded": False}
    }
    
    return {
        "mode": "multi-model",
        "device": device_str,
        "memory_usage": memory_info,
        "models": {
            "medgemma": {
                "loaded": analyzer.model_loaded if analyzer else False,
                "model_id": analyzer.model_id if analyzer else "not initialized"
            },
            "nv_reason_cxr": multi_status.get("nv_reason", {"loaded": False}),
        },
        "all_models_loaded": (
            (analyzer.model_loaded if analyzer else False) and 
            multi_status.get("nv_reason", {}).get("loaded", False)
        ),
        "error": None
    }


@router.post("/load-model")
async def force_load_model():
    """
    Force reload the Med-GEMMA-3 pipeline for recovery from loading failures.
    """
    analyzer = get_analyzer()
    
    # Reset pipeline for clean reload
    analyzer.model = None
    analyzer.processor = None
    analyzer.model_loaded = False
    
    try:
        await analyzer.load_model()
        return {"message": "Model loading initiated", "status": "success"}
    except Exception as e:
        return {"message": f"Model loading failed: {str(e)}", "status": "error"}
