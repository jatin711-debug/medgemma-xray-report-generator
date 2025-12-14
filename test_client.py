import requests
import json
from pathlib import Path
import os

class MedicalImageAnalysisClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.reports_dir = "reports"
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)
    
    def health_check(self):
        response = requests.get(f"{self.base_url}/")
        return response.json()
    
    def analyze_url_and_generate_report(self, image_url: str):
        data = {"image_url": image_url, "image_uuid": "test-uuid-123"}
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{self.base_url}/analyze-url-and-report", json=data, headers=headers)
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            return None
        return response.json()
    
    def list_reports(self):
        response = requests.get(f"{self.base_url}/reports")
        return response.json()
    
    def get_model_status(self):
        response = requests.get(f"{self.base_url}/model-status")
        return response.json()


def example_usage():
    client = MedicalImageAnalysisClient()
    
    # Check model status
    print("=== Model Status ===")
    status = client.get_model_status()
    print(json.dumps(status, indent=2))
    
    # COVID chest X-ray from GitHub (publicly available, no blocking)
    sample_xray_url = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/1-s2.0-S1684118220300608-main.pdf-001.jpg"
    
    print(f"\n--- Generating report from URL ---")
    print(f"URL: {sample_xray_url}")
    
    try:
        url_report = client.analyze_url_and_generate_report(sample_xray_url)
        if url_report:
            print("SUCCESS! Report generated:")
            print(json.dumps(url_report, indent=2))
        else:
            print("Failed to generate report")
    except Exception as e:
        print(f"Error: {e}")
    
    # List reports
    print("\n--- All Reports ---")
    print(client.list_reports())


if __name__ == "__main__":
    example_usage()