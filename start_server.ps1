Write-Host "Starting Medical Image Analysis API..." -ForegroundColor Green
Write-Host ""

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Yellow
pip install -r requirements.txt

# Create reports directory if it doesn't exist
if (!(Test-Path "reports")) {
    New-Item -ItemType Directory -Name "reports"
}

# Start the server
Write-Host ""
Write-Host "Starting FastAPI server..." -ForegroundColor Green
Write-Host "API will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host ""

uvicorn main:app --host 127.0.0.1 --port 8000 --reload