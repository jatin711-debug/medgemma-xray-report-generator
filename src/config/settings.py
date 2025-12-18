"""
Application configuration and settings.
"""

import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    app_name: str = "Medical Image Analysis API"
    app_version: str = "3.0.0"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Model Configuration
    model_id: str = "google/medgemma-4b-it"
    max_tokens: int = 512
    temperature: float = 0.1
    
    # File Storage
    reports_dir: str = "reports"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # Security
    cors_origins: str = "*"  # Configure for production
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    @property
    def allowed_extensions(self) -> List[str]:
        """Allowed image file extensions."""
        return [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"]
    
    @property
    def cors_origins_list(self) -> List[str]:
        """CORS origins as a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]


settings = Settings()

