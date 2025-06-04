@echo off
echo Starting Medical Image Analysis API...
echo.

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Create reports directory if it doesn't exist
if not exist "reports" mkdir reports

REM Start the server
echo.
echo Starting FastAPI server...
echo API will be available at: http://localhost:8000
echo API documentation at: http://localhost:8000/docs
echo.
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause