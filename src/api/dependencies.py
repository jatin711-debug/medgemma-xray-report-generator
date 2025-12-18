"""
API Dependencies and Global State Management.

Provides dependency injection for analyzers and shared state.
"""

import logging
from typing import Optional

from src.core.medical_analyzer import MedicalImageAnalyzer
from src.core.multi_model_analyzer import MultiModelAnalyzer

logger = logging.getLogger(__name__)

# Global analyzer instances
analyzer: Optional[MedicalImageAnalyzer] = None
multi_analyzer: Optional[MultiModelAnalyzer] = None


def get_analyzer() -> MedicalImageAnalyzer:
    """Get the MedicalImageAnalyzer instance."""
    global analyzer
    if analyzer is None:
        analyzer = MedicalImageAnalyzer()
    return analyzer


def get_multi_analyzer() -> MultiModelAnalyzer:
    """Get the MultiModelAnalyzer instance."""
    global multi_analyzer
    if multi_analyzer is None:
        multi_analyzer = MultiModelAnalyzer()
    return multi_analyzer


def set_analyzer(new_analyzer: MedicalImageAnalyzer) -> None:
    """Set the global analyzer instance."""
    global analyzer
    analyzer = new_analyzer


def set_multi_analyzer(new_multi_analyzer: MultiModelAnalyzer) -> None:
    """Set the global multi-model analyzer instance."""
    global multi_analyzer
    multi_analyzer = new_multi_analyzer


async def initialize_analyzers() -> None:
    """Initialize and load all analyzer models."""
    global analyzer, multi_analyzer
    
    logger.info("Initializing Medical Image Analyzers...")
    
    # Initialize MedGemma analyzer
    analyzer = MedicalImageAnalyzer()
    await analyzer.load_model()
    
    # Initialize multi-model analyzer wrapper
    multi_analyzer = MultiModelAnalyzer()
    multi_analyzer.set_medgemma_analyzer(analyzer)
    await multi_analyzer.load_models()
    
    logger.info("All models loaded successfully!")
