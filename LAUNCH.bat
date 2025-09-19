@echo off
echo ================================================================
echo   GovDocShield X Enhanced - Quick Launch Script
echo   Revolutionary Quantum-Neuromorphic Cyber Defense Gateway
echo ================================================================
echo.

echo [1/4] Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.9+ and try again.
    pause
    exit /b 1
)

echo [2/4] Installing dependencies (if needed)...
pip install -q fastapi uvicorn streamlit plotly pandas numpy

echo [3/4] Launching GovDocShield X Enhanced system...
echo.
echo Starting enhanced system in demo mode...
echo Dashboard will open at: http://localhost:8000
echo API available at: http://localhost:8000/api/v2/
echo.

start "" "http://localhost:8000"
python deploy_enhanced.py --mode demo

echo.
echo ================================================================
echo   GovDocShield X Enhanced is now running!
echo   Press Ctrl+C to stop the system
echo ================================================================
pause