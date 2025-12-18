"""
Constants for medical image analysis.

Contains confidence keywords and condition detection patterns used for
parsing AI model responses.
"""

from typing import Dict, List


# Confidence keywords for extracting probability from model text
# Maps language patterns to confidence scores (0.0 - 1.0)
CONFIDENCE_KEYWORDS: Dict[str, float] = {
    # High confidence (80-100%)
    "definite": 0.95, "definitely": 0.95, "clearly": 0.90,
    "confirmed": 0.95, "diagnostic of": 0.92, "classic": 0.90,
    "highly suggestive": 0.88, "strongly suggests": 0.85,
    "typical": 0.85, "characteristic": 0.85,
    
    # Moderate-high confidence (60-80%)
    "consistent with": 0.75, "compatible with": 0.72,
    "suggestive of": 0.70, "likely": 0.72, "probable": 0.75,
    "appears to be": 0.68, "representing": 0.70,
    
    # Moderate confidence (40-60%)
    "possible": 0.50, "may represent": 0.48, "could be": 0.45,
    "cannot exclude": 0.42, "suspicious": 0.55, "questionable": 0.45,
    "might be": 0.45, "potentially": 0.50,
    
    # Low confidence (10-40%)
    "unlikely": 0.25, "doubtful": 0.20, "improbable": 0.15,
    "low probability": 0.20, "less likely": 0.30,
    
    # Negative/Normal (0-10%)
    "no evidence": 0.05, "absent": 0.03, "normal": 0.05,
    "unremarkable": 0.03, "clear lungs": 0.02, "no significant": 0.05,
    "within normal": 0.04, "negative for": 0.03,
}


# Condition-specific keywords that indicate detection
CONDITION_DETECTION_KEYWORDS: Dict[str, List[str]] = {
    "pneumonia": [
        "pneumonia", "consolidation", "infiltrate", 
        "infection", "opacity consistent with infection"
    ],
    "pneumothorax": [
        "pneumothorax", "collapsed lung", 
        "air in pleural", "lung collapse"
    ],
    "tuberculosis": [
        "tuberculosis", "tb", "tuberculous", 
        "cavitary lesion", "miliary pattern"
    ],
}


# MedGemma model configuration
MEDGEMMA_IMAGE_SIZE = 896  # 896x896 resolution
MEDGEMMA_GRID_SIZE = 64    # 64x64 = 4096 tokens
MEDGEMMA_NUM_IMAGE_TOKENS = 4096
MEDGEMMA_PATCH_SIZE = 14   # 14-pixel patches
