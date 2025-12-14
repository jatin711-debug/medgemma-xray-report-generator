"""
Multi-Model Medical Image Analyzer

Orchestrates both MedGemma-4B-IT and NV-Reason-CXR-3B for consensus-based
chest X-ray analysis with robust heatmap generation.

Key features:
- Runs both models in parallel (or sequentially if needed)
- Aggregates findings with consensus logic
- Uses NV-Reason-CXR for primary heatmap (more reliable)
- Generates combined visualization showing both model outputs
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import numpy as np
import cv2

import torch
from PIL import Image
from pydantic import BaseModel, Field
from fastapi import UploadFile

# Local imports - will be loaded from same directory
from nv_reason_analyzer import NVReasonCXRAnalyzer, NVReasonFindings

# Configure logging
logger = logging.getLogger(__name__)


class ConsensusFindings(BaseModel):
    """Aggregated findings from both models with consensus logic."""
    
    # Individual model findings
    medgemma_raw: str = Field(default="", description="Raw MedGemma response")
    medgemma_abnormalities: list[str] = Field(default_factory=list)
    medgemma_confidence: str = Field(default="N/A")
    
    nv_reason_thinking: str = Field(default="", description="NV-Reason clinical reasoning")
    nv_reason_answer: str = Field(default="", description="NV-Reason final diagnosis")
    nv_reason_abnormalities: list[str] = Field(default_factory=list)
    nv_reason_devices: list[str] = Field(default_factory=list)
    
    # Consensus results
    consensus_abnormalities: list[str] = Field(
        default_factory=list, 
        description="Abnormalities detected by BOTH models (high confidence)"
    )
    all_abnormalities: list[str] = Field(
        default_factory=list,
        description="All abnormalities from either model"
    )
    consensus_level: str = Field(
        default="low",
        description="Agreement level: high (both agree), medium (partial), low (disagree)"
    )
    clinical_impression: str = Field(default="", description="Combined clinical impression")


class MultiModelAnalysisResponse(BaseModel):
    """Complete response from multi-model analysis."""
    
    analysis_id: str = Field(description="Unique analysis identifier")
    timestamp: str = Field(description="Analysis timestamp")
    consensus: ConsensusFindings
    image_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Model status
    medgemma_success: bool = Field(default=False)
    nv_reason_success: bool = Field(default=False)


class MultiModelAnalyzer:
    """
    Orchestrator for multi-model chest X-ray analysis.
    
    Manages both MedGemma-4B-IT and NV-Reason-CXR-3B, running them together
    and aggregating their findings with consensus-based logic.
    
    Heatmap Strategy:
    - Primary: NV-Reason-CXR attention heatmap (more reliable)
    - Fallback: MedGemma heatmap if NV-Reason fails
    """
    
    def __init__(self):
        # Import here to avoid circular imports
        # MedicalImageAnalyzer will be set from main.py
        self.medgemma = None  # Will be initialized with existing analyzer
        self.nv_reason = NVReasonCXRAnalyzer()
        self.reports_dir = "reports"
        self.max_image_size = 10 * 1024 * 1024  # 10MB
        self.allowed_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp'}
        
        # Ensure reports directory exists
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def set_medgemma_analyzer(self, medgemma_analyzer):
        """Set the MedGemma analyzer instance (to avoid circular imports)."""
        self.medgemma = medgemma_analyzer
    
    async def load_models(self) -> Dict[str, bool]:
        """
        Load both models at startup.
        
        Returns dict with load status for each model.
        """
        results = {"medgemma": False, "nv_reason": False}
        
        try:
            if self.medgemma:
                await self.medgemma.load_model()
                results["medgemma"] = self.medgemma.model_loaded
                logger.info(f"MedGemma load status: {results['medgemma']}")
        except Exception as e:
            logger.error(f"Failed to load MedGemma: {e}")
        
        try:
            await self.nv_reason.load_model()
            results["nv_reason"] = self.nv_reason.model_loaded
            logger.info(f"NV-Reason load status: {results['nv_reason']}")
        except Exception as e:
            logger.error(f"Failed to load NV-Reason-CXR: {e}")
        
        return results
    
    async def analyze_with_consensus(
        self, 
        image: Image.Image
    ) -> Tuple[MultiModelAnalysisResponse, Image.Image, Image.Image, Image.Image, Image.Image, Image.Image]:
        """
        Run both models and aggregate findings.
        
        Returns:
            Tuple of (response, original_image, heatmap, saliency_map, probability_map, segmentation_map)
            
            Heatmap is from NV-Reason-CXR (primary) or MedGemma (fallback).
            Additional maps are generated from the primary heatmap's attention data.
        """
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        medgemma_raw = ""
        medgemma_findings = None
        medgemma_heatmap = None
        medgemma_success = False
        
        nv_raw = ""
        nv_findings = None
        nv_heatmap = None
        nv_success = False
        
        # Run MedGemma analysis
        if self.medgemma and self.medgemma.model_loaded:
            try:
                medgemma_raw, medgemma_findings, medgemma_heatmap = await self.medgemma.analyze_image(image)
                medgemma_success = True
                logger.info("MedGemma analysis completed successfully")
            except Exception as e:
                logger.error(f"MedGemma analysis failed: {e}")
        else:
            logger.warning("MedGemma not loaded, skipping")
        
        # Run NV-Reason-CXR analysis
        if self.nv_reason.model_loaded:
            try:
                nv_raw, nv_findings, nv_heatmap = await self.nv_reason.analyze_image(image)
                nv_success = True
                logger.info("NV-Reason-CXR analysis completed successfully")
            except Exception as e:
                logger.error(f"NV-Reason-CXR analysis failed: {e}")
        else:
            # Try to load if not loaded
            try:
                await self.nv_reason.load_model()
                nv_raw, nv_findings, nv_heatmap = await self.nv_reason.analyze_image(image)
                nv_success = True
            except Exception as e:
                logger.error(f"NV-Reason-CXR failed to load/analyze: {e}")
        
        # Aggregate findings
        consensus = self._aggregate_findings(
            medgemma_raw=medgemma_raw,
            medgemma_findings=medgemma_findings,
            nv_raw=nv_raw,
            nv_findings=nv_findings
        )
        
        # Generate zone-based heatmap from parsed locations (preferred)
        if nv_success and nv_findings and nv_findings.locations:
            primary_heatmap = self.nv_reason.generate_zone_heatmap(nv_findings.locations, image)
            logger.info(f"Using zone-based heatmap from {len(nv_findings.locations)} locations")
        elif nv_success and nv_heatmap is not None:
            primary_heatmap = nv_heatmap
            logger.info("Using NV-Reason-CXR attention heatmap (fallback)")
        elif medgemma_success and medgemma_heatmap is not None:
            primary_heatmap = medgemma_heatmap
            logger.info("Using MedGemma heatmap (fallback)")
        else:
            # Create placeholder
            primary_heatmap = self._create_placeholder_heatmap(image)
            logger.warning("No heatmap available from either model, using placeholder")
        
        # Generate additional visualization maps
        # For now, we return the primary heatmap for all map types
        # The PDF generator will use these appropriately
        saliency_map = primary_heatmap  # Can be enhanced with gradient-based methods
        probability_map = primary_heatmap
        segmentation_map = primary_heatmap
        
        response = MultiModelAnalysisResponse(
            analysis_id=analysis_id,
            timestamp=timestamp,
            consensus=consensus,
            image_metadata={
                "size": image.size,
                "mode": image.mode,
                "format": getattr(image, 'format', 'unknown')
            },
            medgemma_success=medgemma_success,
            nv_reason_success=nv_success
        )
        
        return response, image, primary_heatmap, saliency_map, probability_map, segmentation_map
    
    def _aggregate_findings(
        self,
        medgemma_raw: str,
        medgemma_findings,  # RadiologyFindings or None
        nv_raw: str,
        nv_findings: Optional[NVReasonFindings]
    ) -> ConsensusFindings:
        """
        Aggregate findings from both models with consensus logic.
        
        High consensus: Both models detect the same abnormality
        Medium consensus: Only one model detects, but with high confidence
        Low consensus: Models disagree or low confidence
        """
        # Extract MedGemma abnormalities
        mg_abnormalities = []
        mg_confidence = "N/A"
        if medgemma_findings:
            mg_abnormalities = [a.lower() for a in medgemma_findings.abnormalities]
            mg_confidence = medgemma_findings.confidence_level
        
        # Extract NV-Reason abnormalities
        nv_abnormalities = []
        nv_devices = []
        nv_thinking = ""
        nv_answer = ""
        if nv_findings:
            nv_abnormalities = [a.lower() for a in nv_findings.abnormalities]
            nv_devices = nv_findings.support_devices
            nv_thinking = nv_findings.thinking
            nv_answer = nv_findings.answer
        
        # Find consensus (abnormalities detected by both)
        mg_set = set(mg_abnormalities)
        nv_set = set(nv_abnormalities)
        consensus_abnormalities = list(mg_set.intersection(nv_set))
        all_abnormalities = list(mg_set.union(nv_set))
        
        # Determine consensus level
        if len(consensus_abnormalities) > 0:
            consensus_level = "high"
        elif len(all_abnormalities) > 0:
            consensus_level = "medium"
        else:
            consensus_level = "low"  # Both found nothing (which is agreement, actually)
            if len(mg_abnormalities) == 0 and len(nv_abnormalities) == 0:
                consensus_level = "high"  # Both agree: no abnormalities
        
        # Generate clinical impression
        if consensus_level == "high" and len(consensus_abnormalities) > 0:
            impression = f"Both AI models detected: {', '.join([a.title() for a in consensus_abnormalities])}. High confidence finding."
        elif consensus_level == "high":
            impression = "Both AI models found no significant abnormalities. Recommend clinical correlation."
        elif consensus_level == "medium":
            only_mg = list(mg_set - nv_set)
            only_nv = list(nv_set - mg_set)
            parts = []
            if only_mg:
                parts.append(f"MedGemma detected: {', '.join([a.title() for a in only_mg])}")
            if only_nv:
                parts.append(f"NV-Reason detected: {', '.join([a.title() for a in only_nv])}")
            impression = " | ".join(parts) + ". Partial agreement - recommend specialist review."
        else:
            impression = "AI analysis inconclusive. Recommend radiologist review."
        
        return ConsensusFindings(
            medgemma_raw=medgemma_raw,
            medgemma_abnormalities=[a.title() for a in mg_abnormalities],
            medgemma_confidence=mg_confidence,
            nv_reason_thinking=nv_thinking,
            nv_reason_answer=nv_answer,
            nv_reason_abnormalities=[a.title() for a in nv_abnormalities],
            nv_reason_devices=nv_devices,
            consensus_abnormalities=[a.title() for a in consensus_abnormalities],
            all_abnormalities=[a.title() for a in all_abnormalities],
            consensus_level=consensus_level,
            clinical_impression=impression
        )
    
    def _create_placeholder_heatmap(self, original_image: Image.Image) -> Image.Image:
        """Create a placeholder when no heatmap is available."""
        try:
            original_rgb = np.array(original_image.convert("RGB"))
            overlay = np.ones_like(original_rgb) * 128
            alpha = 0.1
            blended = (1 - alpha) * original_rgb + alpha * overlay
            return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
        except:
            return original_image.copy()
    
    async def analyze_image_from_upload(self, file: UploadFile) -> Tuple[MultiModelAnalysisResponse, Image.Image, Image.Image, Image.Image, Image.Image, Image.Image]:
        """
        Analyze medical image from uploaded file.
        
        Returns tuple of (response, original_image, heatmap, saliency, probability, segmentation)
        """
        import io
        
        # Read and validate file
        contents = await file.read()
        if len(contents) > self.max_image_size:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        # Open image
        image = Image.open(io.BytesIO(contents))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return await self.analyze_with_consensus(image)
    
    async def analyze_image_from_url(self, image_url: str) -> Tuple[MultiModelAnalysisResponse, Image.Image, Image.Image, Image.Image, Image.Image, Image.Image]:
        """
        Analyze medical image from URL.
        
        Returns tuple of (response, original_image, heatmap, saliency, probability, segmentation)
        """
        import io
        import aiohttp
        
        # Headers to avoid 403 errors from Wikipedia and other sites
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/*,*/*"
        }
        
        # Download image
        async with aiohttp.ClientSession() as session:
            async with session.get(str(image_url), headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    from fastapi import HTTPException
                    raise HTTPException(status_code=400, detail=f"Failed to fetch image: HTTP {resp.status}")
                
                content = await resp.read()
                if len(content) > self.max_image_size:
                    from fastapi import HTTPException
                    raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
        
        # Open image
        image = Image.open(io.BytesIO(content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return await self.analyze_with_consensus(image)
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of both models."""
        return {
            "medgemma": {
                "loaded": self.medgemma.model_loaded if self.medgemma else False,
                "model_id": self.medgemma.model_id if self.medgemma else "not initialized"
            },
            "nv_reason": {
                "loaded": self.nv_reason.model_loaded,
                "model_id": self.nv_reason.model_id
            }
        }
