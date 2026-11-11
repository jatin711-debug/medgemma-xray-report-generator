"""Command-line launcher for the experimental MedGemma API."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Run the experimental MedGemma API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    quantization = parser.add_mutually_exclusive_group()
    quantization.add_argument("--quantize", dest="quantize", action="store_const", const="1")
    quantization.add_argument("--no-quantize", dest="quantize", action="store_const", const="0")
    parser.set_defaults(quantize=None)
    args = parser.parse_args()
    if args.quantize is not None:
        os.environ["MEDGEMMA_QUANTIZE"] = args.quantize

    import uvicorn
    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
