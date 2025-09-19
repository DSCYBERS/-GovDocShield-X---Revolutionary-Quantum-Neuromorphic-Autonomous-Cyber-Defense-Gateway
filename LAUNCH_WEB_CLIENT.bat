@echo off
echo ================================================================
echo   GovDocShield X Enhanced - Web Client Launcher
echo   Modern Web-Based Interface for Cyber Defense Gateway
echo ================================================================
echo.

echo [1/3] Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.9+ and try again.
    pause
    exit /b 1
)

echo [2/3] Installing web client dependencies...
pip install -q fastapi uvicorn websockets

echo [3/3] Starting web client server...
echo.
echo 🌐 Web Interface: http://localhost:8001
echo 📱 Mobile-Friendly: Responsive design for all devices
echo 🔌 Real-time Updates: WebSocket-powered live monitoring
echo 🎛️ Full Control: Complete system management interface
echo.

echo Opening web interface...
timeout /t 3 /nobreak >nul
start "" "http://localhost:8001"

echo Starting server...
python web_client\server.py

echo.
echo ================================================================
echo   Web client server stopped
echo ================================================================
pause