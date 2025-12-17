"""
MedGemma Medical Image Analyzer

Single-model chest X-ray analysis using MedGemma-4B-IT (full precision).
Simplified architecture for better gradient extraction and future GradCAM support.
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

# Configure logging
logger = logging.getLogger(__name__)


class ConsensusFindings(BaseModel):
    """Findings from MedGemma analysis (kept for backwards compatibility)."""
    
    # MedGemma findings
    medgemma_raw: str = Field(default="", description="Raw MedGemma response")
    medgemma_abnormalities: list[str] = Field(default_factory=list)
    medgemma_confidence: str = Field(default="N/A")
    
    # All abnormalities (same as medgemma for single-model)
    all_abnormalities: list[str] = Field(
        default_factory=list,
        description="All detected abnormalities"
    )
    clinical_impression: str = Field(default="", description="Clinical impression")


class MultiModelAnalysisResponse(BaseModel):
    """Complete response from analysis (kept for backwards compatibility)."""
    
    analysis_id: str = Field(description="Unique analysis identifier")
    timestamp: str = Field(description="Analysis timestamp")
    consensus: ConsensusFindings
    image_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Model status
    medgemma_success: bool = Field(default=False)


class MultiModelAnalyzer:
    """
    MedGemma-only chest X-ray analyzer.
    
    Uses MedGemma-4B-IT without quantization for better gradient extraction.
    Class name kept as MultiModelAnalyzer for backwards compatibility.
    """
    
    def __init__(self):
        self.medgemma = None  # Will be initialized with existing analyzer
        self.gradcam = None   # GradCAM analyzer (initialized after model load)
        self.reports_dir = "reports"
        self.max_image_size = 10 * 1024 * 1024  # 10MB
        self.allowed_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp'}
        
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def set_medgemma_analyzer(self, medgemma_analyzer):
        """Set the MedGemma analyzer instance."""
        self.medgemma = medgemma_analyzer
    
    async def load_models(self) -> Dict[str, bool]:
        """Load MedGemma model at startup."""
        results = {"medgemma": False}
        
        try:
            if self.medgemma:
                await self.medgemma.load_model()
                results["medgemma"] = self.medgemma.model_loaded
                logger.info(f"MedGemma load status: {results['medgemma']}")
                
                # Initialize GradCAM after model is loaded
                if self.medgemma.model_loaded:
                    try:
                        from gradcam_analyzer import MedGemmaGradCAM
                        self.gradcam = MedGemmaGradCAM(
                            self.medgemma.model, 
                            self.medgemma.processor
                        )
                        logger.info("✅ GradCAM initialized successfully")
                    except Exception as e:
                        logger.warning(f"Could not initialize GradCAM: {e}")
                        self.gradcam = None
        except Exception as e:
            logger.error(f"Failed to load MedGemma: {e}")
        
        return results
    
    async def analyze_with_consensus(
        self, 
        image: Image.Image
    ) -> Tuple[MultiModelAnalysisResponse, Image.Image, Image.Image, Image.Image, 
               Image.Image, Image.Image, Optional[Image.Image], list, Dict[str, float]]:
        """
        Run MedGemma analysis and generate visualizations.
        
        Returns:
            Tuple of (response, original_image, heatmap, saliency_map, probability_map, 
                      segmentation_map, localization_map, detected_regions, condition_confidences)
        """
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        medgemma_raw = ""
        medgemma_findings = None
        medgemma_heatmap = None
        medgemma_raw_heatmap_np = None
        medgemma_success = False
        
        # Run MedGemma analysis
        if self.medgemma and self.medgemma.model_loaded:
            try:
                medgemma_raw, medgemma_findings, medgemma_heatmap = await self.medgemma.analyze_image(image)
                medgemma_success = True
                logger.info("MedGemma analysis completed successfully")
                
                # 🔍 Log what MedGemma outputs
                logger.info("=" * 60)
                logger.info("📋 MEDGEMMA RAW OUTPUT:")
                logger.info(f"{medgemma_raw[:500]}..." if len(medgemma_raw) > 500 else medgemma_raw)
                logger.info("-" * 40)
                if medgemma_findings:
                    logger.info(f"📊 PARSED FINDINGS:")
                    logger.info(f"   Abnormalities: {medgemma_findings.abnormalities}")
                    logger.info(f"   Confidence: {medgemma_findings.confidence_level}")
                    logger.info(f"   Impression: {medgemma_findings.impression}")
                logger.info("=" * 60)
                
                # Get raw heatmap for localization
                if hasattr(self.medgemma, '_last_raw_heatmap'):
                    medgemma_raw_heatmap_np = self.medgemma._last_raw_heatmap
                    logger.info(f"Retrieved raw heatmap: shape={medgemma_raw_heatmap_np.shape if medgemma_raw_heatmap_np is not None else 'None'}")
            except Exception as e:
                logger.error(f"MedGemma analysis failed: {e}")
        else:
            logger.warning("MedGemma not loaded")
        
        # Build consensus findings
        consensus = self._aggregate_findings(
            medgemma_raw=medgemma_raw,
            medgemma_findings=medgemma_findings
        )
        
        # Extract condition confidences first (needed for GradCAM)
        condition_confidences = {}
        if consensus:
            abnormalities_text = consensus.all_abnormalities
            
            TRACKED_CONDITIONS = {
                "pneumonia": ["pneumonia", "consolidation", "infiltrate"],
                "pneumothorax": ["pneumothorax", "ptx"],
                "tuberculosis": ["tuberculosis", "tb"],
            }
            
            # Negative indicators - if present with keyword, it's NOT detected
            NEGATIVE_PHRASES = ["no evidence", "not detected", "no sign", "negative for", 
                               "absent", "no ", "without", "unremarkable", "clear"]
            
            def is_condition_detected(abnorm_list, keywords):
                """Check if condition is positively detected (not negated)."""
                for abnorm in abnorm_list:
                    abnorm_lower = abnorm.lower()
                    
                    for keyword in keywords:
                        if keyword in abnorm_lower:
                            # Check if this is a NEGATIVE mention
                            is_negative = any(neg in abnorm_lower for neg in NEGATIVE_PHRASES)
                            if not is_negative:
                                return True
                return False
            
            for condition_name, keywords in TRACKED_CONDITIONS.items():
                detected = is_condition_detected(abnormalities_text, keywords)
                condition_confidences[condition_name] = 0.75 if detected else 0.08
            
            logger.info(f"Abnormalities: {abnormalities_text}")
            logger.info(f"Extracted condition confidences: {condition_confidences}")
        
        # =========================================================================
        # HEATMAP GENERATION: GradCAM (preferred) or Attention-based (fallback)
        # =========================================================================
        gradcam_heatmap = None
        gradcam_raw = None
        gradcam_regions = []
        
        # Try GradCAM first (condition-specific evidence map)
        if self.gradcam is not None and medgemma_success:
            try:
                # Use the highest-confidence condition for GradCAM
                if condition_confidences:
                    top_condition = max(condition_confidences.items(), key=lambda x: x[1])
                    target_condition = top_condition[0]
                else:
                    target_condition = "pneumonia"  # Default
                
                logger.info(f"🎯 Generating clinical GradCAM for: {target_condition}")
                # 4 returns: (ai_evidence_map, localization_map, raw_cam, regions)
                gradcam_heatmap, gradcam_localization, gradcam_raw, gradcam_regions = self.gradcam.generate_cam(image, target_condition)
                
                if gradcam_heatmap is not None:
                    logger.info(f"✅ Clinical GradCAM: {len(gradcam_regions)} regions detected")
            except Exception as e:
                logger.warning(f"GradCAM generation failed, falling back to attention: {e}")
                gradcam_heatmap = None
                gradcam_localization = None
                gradcam_regions = []
        
        # Choose primary heatmap: GradCAM > Attention > Placeholder
        if gradcam_heatmap is not None:
            primary_heatmap = gradcam_heatmap  # Pure heatmap (no contours)
            logger.info("✅ Using GradCAM AI Evidence Map")
            # Update raw heatmap for localization if available
            if gradcam_raw is not None:
                medgemma_raw_heatmap_np = gradcam_raw
        elif medgemma_success and medgemma_heatmap is not None:
            primary_heatmap = medgemma_heatmap
            logger.info("⚠️ Using attention-based heatmap (GradCAM unavailable)")
        else:
            primary_heatmap = self._create_placeholder_heatmap(image)
            logger.warning("No heatmap available, using placeholder")
        
        # Generate saliency map (density-based - kept for reference)
        if self.medgemma is not None:
            saliency_map = self.medgemma._generate_density_saliency_map(image)
            logger.info("Generated density-based saliency map")
        else:
            saliency_map = primary_heatmap
        
        # Generate localization map
        # If GradCAM already extracted regions, use the separate localization image
        localization_map = None
        detected_regions = []
        
        if gradcam_regions and gradcam_localization is not None:
            # GradCAM already created localization map with contours
            localization_map = gradcam_localization  # Separate image with contours
            detected_regions = gradcam_regions
            logger.info(f"Using GradCAM localization map: {len(detected_regions)} regions")
        elif self.medgemma is not None and medgemma_raw_heatmap_np is not None:
            # Fallback to old localization method
            localization_map, detected_regions = self.medgemma._generate_localization_map(
                medgemma_raw_heatmap_np, image
            )
            logger.info(f"Generated localization map with {len(detected_regions)} regions")
        
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
            medgemma_success=medgemma_success
        )
        
        return (response, image, primary_heatmap, saliency_map, probability_map, 
                segmentation_map, localization_map, detected_regions, condition_confidences)
    
    def _aggregate_findings(
        self,
        medgemma_raw: str,
        medgemma_findings  # RadiologyFindings or None
    ) -> ConsensusFindings:
        """Build findings from MedGemma analysis."""
        mg_abnormalities = []
        mg_confidence = "N/A"
        clinical_impression = ""
        
        if medgemma_findings:
            mg_abnormalities = [a for a in medgemma_findings.abnormalities]
            mg_confidence = medgemma_findings.confidence_level
            clinical_impression = medgemma_findings.impression
        
        # Build clinical impression if not provided
        if not clinical_impression:
            if mg_abnormalities:
                clinical_impression = f"MedGemma detected: {', '.join([a.title() for a in mg_abnormalities])}."
            else:
                clinical_impression = "No significant abnormalities detected by MedGemma analysis."
        
        return ConsensusFindings(
            medgemma_raw=medgemma_raw,
            medgemma_abnormalities=[a.title() for a in mg_abnormalities],
            medgemma_confidence=mg_confidence,
            all_abnormalities=[a.title() for a in mg_abnormalities],
            clinical_impression=clinical_impression
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
    
    async def analyze_image_from_upload(self, file: UploadFile):
        """Analyze medical image from uploaded file."""
        import io
        
        contents = await file.read()
        if len(contents) > self.max_image_size:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        image = Image.open(io.BytesIO(contents))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return await self.analyze_with_consensus(image)
    
    async def analyze_image_from_url(self, image_url: str):
        """Analyze medical image from URL."""
        import io
        import aiohttp
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "image/*,*/*"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(str(image_url), headers=headers, 
                                   timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    from fastapi import HTTPException
                    raise HTTPException(status_code=400, detail=f"Failed to fetch image: HTTP {resp.status}")
                
                content = await resp.read()
                if len(content) > self.max_image_size:
                    from fastapi import HTTPException
                    raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
        
        image = Image.open(io.BytesIO(content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return await self.analyze_with_consensus(image)
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of MedGemma model."""
        return {
            "medgemma": {
                "loaded": self.medgemma.model_loaded if self.medgemma else False,
                "model_id": self.medgemma.model_id if self.medgemma else "not initialized"
            }
        }
