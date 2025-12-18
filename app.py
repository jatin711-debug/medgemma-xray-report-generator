"""
MedGemma Report Generator - Entry Point

Medical Image Analysis API using MedGemma-4B-IT.

Usage:
    python app.py           # Run with full precision (bfloat16)
    python app.py -q        # Run with 4-bit quantization (reduced VRAM)
    python app.py --port 8080  # Run on custom port
"""

# Suppress TensorFlow/oneDNN warnings (must be set before ANY imports)
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress all TF logs
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "0"
    
import sys
import signal
import argparse


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
    """Create the FastAPI app."""
    from src.api.app import create_app
    return create_app()


def graceful_shutdown(signum, frame):
    """Handle graceful shutdown on Ctrl+C."""
    print("\n👋 Shutting down gracefully...")
    sys.exit(0)


# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)

# Check if quantization is enabled via environment variable
# This persists across uvicorn reloads
if os.environ.get("MEDGEMMA_QUANTIZE") == "1":
    pass  # Already set by main block
elif __name__ == "__main__":
    # Only parse args when running directly
    args = parse_args()
    if args.quantize:
        os.environ["MEDGEMMA_QUANTIZE"] = "1"
        print("🔧 4-bit quantization ENABLED (reduced VRAM mode)")
    else:
        os.environ.setdefault("MEDGEMMA_QUANTIZE", "0")

# Create the FastAPI application
app = create_application()


if __name__ == "__main__":
    import uvicorn
    
    args = parse_args()
    
    # Set quantization env var before uvicorn starts workers
    if args.quantize:
        os.environ["MEDGEMMA_QUANTIZE"] = "1"
    
    print(f"🚀 Starting MedGemma API on {args.host}:{args.port}")
    print(f"   Quantization: {'4-bit' if args.quantize else 'Full precision (bfloat16)'}")
    print(f"   Press Ctrl+C to stop")
    
    try:
        uvicorn.run(
            "app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
