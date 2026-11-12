"""Small client for the experimental API."""
import argparse
import json
import mimetypes
import os
from pathlib import Path

import requests


class MedicalImageAnalysisClient:
    def __init__(self, base_url="http://127.0.0.1:8000", api_key=None):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["X-API-Key"] = api_key or os.getenv("MEDGEMMA_API_KEY", "")

    def health_check(self):
        response = self.session.get(f"{self.base_url}/", timeout=30)
        response.raise_for_status()
        return response.json()

    def get_model_status(self):
        response = self.session.get(f"{self.base_url}/model-status", timeout=30)
        response.raise_for_status()
        return response.json()

    def list_reports(self):
        response = self.session.get(f"{self.base_url}/reports", timeout=30)
        response.raise_for_status()
        return response.json()

    def analyze_and_generate_report(self, image_path):
        path = Path(image_path)
        with path.open("rb") as image:
            response = self.session.post(f"{self.base_url}/analyze-and-report",
                                     files={"file": (path.name, image, mimetypes.guess_type(path.name)[0] or "application/octet-stream")}, timeout=600)
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a chest X-ray to the experimental API")
    parser.add_argument("image", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    print(json.dumps(MedicalImageAnalysisClient(args.url).analyze_and_generate_report(args.image), indent=2))
