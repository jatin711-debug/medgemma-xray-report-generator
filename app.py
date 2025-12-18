"""
MedGemma Report Generator - Entry Point

Medical Image Analysis API using MedGemma-4B-IT.
"""

import uvicorn
from src.api.app import create_app

# Create the FastAPI application
app = create_app()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
