import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    app_name: str = "Medical Image Analysis API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Model Configuration
    model_id: str = "google/medgemma-1.5-4b-it"
    max_tokens: int = 512
    temperature: float = 0.1
    
    # File Storage
    reports_dir: str = "reports"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: list = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
    
    # Security
    cors_origins: list = ["*"]  # Configure for production
    
    class Config:
        env_file = ".env"

settings = Settings()