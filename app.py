"""
MedGemma Report Generator - Entry Point

Medical Image Analysis API using MedGemma-4B-IT.

Usage:
    python app.py           # Run with full precision (bfloat16)
    python app.py -q        # Run with 4-bit quantization (reduced VRAM)
    python app.py --port 8080  # Run on custom port
"""

import argparse
import os
import uvicorn


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MedGemma Medical Image Analysis API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py -q              Enable 4-bit quantization (saves ~50% VRAM)
  python app.py --port 8080     Run on port 8080
  python app.py -q --reload     Enable quantization + hot reload
        """
    )
    
    parser.add_argument(
        "-q", "--quantize",
        action="store_true",
        help="Enable 4-bit quantization for reduced VRAM usage (~4GB vs ~8GB)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable hot reload for development"
    )
    
    return parser.parse_args()


def create_application():
    """Create the FastAPI app with configuration from args."""
    from src.api.app import create_app
    return create_app()


# Parse arguments early to set environment variables before imports
args = parse_args() if __name__ == "__main__" else argparse.Namespace(quantize=False)

# Set quantization environment variable for other modules to read
if hasattr(args, 'quantize') and args.quantize:
    os.environ["MEDGEMMA_QUANTIZE"] = "1"
    print("🔧 4-bit quantization ENABLED (reduced VRAM mode)")
else:
    os.environ["MEDGEMMA_QUANTIZE"] = "0"

# Create the FastAPI application
app = create_application()


if __name__ == "__main__":
    print(f"🚀 Starting MedGemma API on {args.host}:{args.port}")
    print(f"   Quantization: {'4-bit' if args.quantize else 'Full precision (bfloat16)'}")
    
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

