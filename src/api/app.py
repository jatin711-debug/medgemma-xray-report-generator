"""
FastAPI Application Factory.

Creates and configures the FastAPI application with lifespan management.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config.settings import settings
from src.api.dependencies import initialize_analyzers
from src.api.routes import router

os.environ['TORCH_COMPILE_DISABLE'] = '1'

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Medical Image Analysis API (MedGemma-4B-IT)...")
    await initialize_analyzers()
    yield
    # Shutdown
    logger.info("Shutting down Medical Image Analysis API...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title=settings.app_name,
        description="AI-powered medical image analysis using MedGemma-4B-IT (Full Precision)",
        version=settings.app_version,
        lifespan=lifespan
    )
    
    # Include API routes
    app.include_router(router)
    
    return app
