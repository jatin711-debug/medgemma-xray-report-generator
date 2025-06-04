# Medical Image Analysis API

A production-level FastAPI backend for medical image analysis using Google's MedGemma model. This API provides structured radiology reports and PDF generation capabilities.

## Features

- 🏥 Medical image analysis using Google MedGemma-4B-IT model
- 📊 Structured radiology findings with confidence levels
- 📄 Automatic PDF report generation
- 🔗 RESTful API endpoints
- 📝 Comprehensive logging and error handling
- 🐳 Docker support for deployment

## Windows Setup Instructions

### Prerequisites

1. **Python 3.8+** - Download from [python.org](https://www.python.org/downloads/)
2. **Git** (optional) - Download from [git-scm.com](https://git-scm.com/)
3. **CUDA-compatible GPU** (recommended for faster inference)

### Quick Start (Windows)

1. **Clone or download this project**
2. **Open Command Prompt or PowerShell** in the project directory
3. **Run the setup script:**

   **Option A: Using Batch File (Command Prompt)**
   ```cmd
   start_server.bat
   ```

   **Option B: Using PowerShell**
   ```powershell
   .\start_server.ps1
   ```

4. **Access the API:**
   - API: http://localhost:8000
   - Interactive Documentation: http://localhost:8000/docs
   - Alternative Docs: http://localhost:8000/redoc

### Manual Setup (Windows)

If you prefer manual setup:

```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Create reports directory
mkdir reports

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### 1. Health Check
```
GET /
```
Returns API status and health information.

### 2. Analyze Image
```
POST /analyze
```
Upload a medical image and get structured analysis results.

**Example using curl:**
```cmd
curl -X POST "http://localhost:8000/analyze" -F "file=@path\to\your\image.jpg"
```

### 3. Analyze and Generate Report
```
POST /analyze-and-report
```
Upload an image, analyze it, and generate a PDF report.

### 4. Download Report
```
GET /download-report/{analysis_id}
```
Download a generated PDF report by analysis ID.

### 5. List Reports
```
GET /reports
```
Get a list of all generated reports.

## Using the Test Client

A test client is provided in `test_client.py`. To use it:

```python
from test_client import MedicalImageAnalysisClient

client = MedicalImageAnalysisClient()

# Analyze an image
result = client.analyze_image("path/to/your/image.jpg")
print(result)

# Generate a report
report = client.analyze_and_generate_report("path/to/your/image.jpg")
print(f"Report generated: {report}")
```

## Response Format

The API returns structured responses with the following format:

```json
{
  "analysis_id": "uuid",
  "timestamp": "2025-06-04T10:30:00",
  "findings": {
    "anatomical_structures": ["list of identified structures"],
    "abnormalities": ["list of detected abnormalities"],
    "impression": "clinical impression summary",
    "recommendations": ["list of recommendations"],
    "confidence_level": "High/Medium/Low"
  },
  "raw_response": "complete AI model response",
  "image_metadata": {
    "filename": "image.jpg",
    "size": 1024000,
    "format": "JPEG",
    "dimensions": [512, 512]
  }
}
```

## PDF Reports

Generated PDF reports include:
- Analysis metadata (ID, timestamp, confidence)
- Structured findings sections
- Clinical impression
- Recommendations
- Complete AI analysis text

Reports are saved in the `reports/` directory with the naming pattern: `medical_report_{analysis_id}.pdf`

## Production Deployment

### Docker (Recommended)

```cmd
# Build the image
docker build -t medical-image-api .

# Run the container
docker run -p 8000:8000 -v %cd%\reports:/app/reports medical-image-api
```

### Environment Configuration

Copy `.env.example` to `.env` and modify settings as needed:

```bash
cp .env.example .env
```

## Hardware Requirements

- **Minimum:** 8GB RAM, CPU-only inference
- **Recommended:** 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM
- **Storage:** 10GB+ for model files and reports

## Troubleshooting

### Common Issues on Windows

1. **PowerShell Execution Policy Error:**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

2. **CUDA/GPU Issues:**
   - Install NVIDIA drivers and CUDA toolkit
   - Or set `device_map="cpu"` in config for CPU-only inference

3. **Port Already in Use:**
   - Change port in config.py or use: `uvicorn main:app --port 8001`

4. **Large Model Download:**
   - First run downloads ~8GB model files
   - Ensure stable internet connection

### Performance Tips

- Use GPU acceleration when available
- Adjust `max_tokens` in config for shorter/longer responses
- Monitor memory usage with large batch processing

## Security Considerations

For production deployment:
- Change default CORS settings
- Add authentication/authorization
- Use HTTPS
- Implement rate limiting
- Validate file uploads thoroughly

## License

This project is for educational and research purposes. Please review Google's MedGemma license terms for commercial use.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the interactive API documentation at `/docs`
3. Check logs for detailed error information