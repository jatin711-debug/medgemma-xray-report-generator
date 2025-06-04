import requests
import json
from pathlib import Path
import os

class MedicalImageAnalysisClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        # Ensure reports directory exists for downloaded files
        self.reports_dir = "reports"
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)
    
    def health_check(self):
        """Check API health"""
        response = requests.get(f"{self.base_url}/")
        return response.json()
    
    def analyze_image(self, image_path: str):
        """Analyze a medical image file"""
        with open(image_path, 'rb') as image_file:
            files = {'file': image_file}
            response = requests.post(f"{self.base_url}/analyze", files=files)
            return response.json()
    
    def analyze_image_url(self, image_url: str):
        """Analyze a medical image from URL"""
        data = {"image_url": image_url}
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{self.base_url}/analyze-url", json=data, headers=headers)
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            return None
        return response.json()
    
    def analyze_and_generate_report(self, image_path: str):
        """Analyze image file and generate PDF report"""
        with open(image_path, 'rb') as image_file:
            files = {'file': image_file}
            response = requests.post(f"{self.base_url}/analyze-and-report", files=files)
            return response.json()
    
    def analyze_url_and_generate_report(self, image_url: str):
        """Analyze image from URL and generate PDF report"""
        data = {"image_url": image_url}
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{self.base_url}/analyze-url-and-report", json=data, headers=headers)
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            return None
        return response.json()
    
    def download_report(self, analysis_id: str, save_path: str = None):
        """Download PDF report"""
        response = requests.get(f"{self.base_url}/download-report/{analysis_id}")
        
        if response.status_code == 200:
            if not save_path:
                # Save to reports directory instead of root
                save_path = os.path.join(self.reports_dir, f"downloaded_report_{analysis_id}.pdf")
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return save_path
        else:
            return None
    
    def list_reports(self):
        """List all generated reports"""
        response = requests.get(f"{self.base_url}/reports")
        return response.json()
    
    def get_model_status(self):
        """Get current model loading status"""
        response = requests.get(f"{self.base_url}/model-status")
        return response.json()
    
    def force_reload_model(self):
        """Force reload the model (useful if loading failed)"""
        response = requests.post(f"{self.base_url}/load-model")
        return response.json()

def example_usage():
    """Example usage of the client with both file uploads and URLs"""
    client = MedicalImageAnalysisClient()
    
    # Health check
    # print("Health check:", client.health_check())
    
    # Check model status
    # print("Model status:", client.get_model_status())
    
    # If model failed to load, try to reload it
    status = client.get_model_status()
    if not status.get('loaded', False) and status.get('error'):
        print("Model not loaded, attempting to reload...")
        reload_result = client.force_reload_model()
        print("Reload result:", reload_result)
    
    # Example with a sample image file (you'll need to provide an actual image path)
    # image_path = "sample_xray.jpg"
    # 
    # # Analyze image file only
    # analysis = client.analyze_image(image_path)
    # print("File Analysis result:", json.dumps(analysis, indent=2))
    
    # Example with HTTPS URL (using a medical image URL)
    sample_xray_url = "https://upload.wikimedia.org/wikipedia/commons/5/57/Pneumothorax_CT.jpg"
    
    try:
        # Analyze image from URL
        # print(f"\n--- Analyzing image from URL: {sample_xray_url} ---")
        # url_analysis = client.analyze_image_url(sample_xray_url)
        # print("URL Analysis result:", json.dumps(url_analysis, indent=2))
        
        # Generate report from URL
        print(f"\n--- Generating report from URL ---")
        url_report = client.analyze_url_and_generate_report(sample_xray_url)
        print("URL Report generated:", url_report)
        
        # Download the report
        # analysis_id = url_report['analysis_id']
        # pdf_path = client.download_report(analysis_id)
        # print(f"Report downloaded to: {pdf_path}")
        
    except Exception as e:
        print(f"Error with URL analysis: {e}")
    
    # List all reports
    try:
        reports = client.list_reports()
        print("\nAll reports:", reports)
    except Exception as e:
        print(f"Error listing reports: {e}")

if __name__ == "__main__":
    example_usage()