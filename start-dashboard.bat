@echo off
echo 🛡️ GovDocShield X Dashboard Launcher
echo =====================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo 📦 Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Install dashboard dependencies if needed
echo 📦 Checking dashboard dependencies...
pip install -r requirements-dashboard.txt --quiet

REM Launch dashboard
echo 🚀 Starting GovDocShield X Dashboard...
echo 🌐 Opening in browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard
echo.

python dashboard.py

pause