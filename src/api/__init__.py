# API Package
"""FastAPI application, routes, and request/response schemas."""

from .schemas import (
    ImageURLRequest,
    ImageURLReportRequest,
    RadiologyFindings,
    AnalysisResponse,
    PDFGenerationResponse,
    ModelStatus,
)

# Lazy import to avoid circular dependency
def create_app():
    from .app import create_app as _create_app
    return _create_app()

__all__ = [
    "create_app",
    "ImageURLRequest",
    "ImageURLReportRequest",
    "RadiologyFindings",
    "AnalysisResponse",
    "PDFGenerationResponse",
    "ModelStatus",
]
