# GovDocShield X Installation Script for Windows PowerShell
# Run this script to set up the CLI environment

Write-Host "🛡️ GovDocShield X Installation Script" -ForegroundColor Blue
Write-Host "Setting up quantum-neuromorphic threat detection CLI..." -ForegroundColor Cyan

# Check Python installation
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

# Check if virtual environment exists
if (Test-Path "venv") {
    Write-Host "📦 Virtual environment already exists" -ForegroundColor Yellow
} else {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Cyan
    python -m venv venv
}

# Activate virtual environment
Write-Host "🔄 Activating virtual environment..." -ForegroundColor Cyan
& ".\venv\Scripts\Activate.ps1"

# Install CLI dependencies
Write-Host "📦 Installing CLI dependencies..." -ForegroundColor Cyan
pip install -r requirements-cli.txt

# Make govdocshield command available
Write-Host "🔧 Setting up govdocshield command..." -ForegroundColor Cyan

# Create batch file for Windows
$batchContent = @"
@echo off
cd /d "%~dp0"
call venv\Scripts\activate
python govdocshield.py %*
"@

$batchContent | Out-File -FilePath "govdocshield.bat" -Encoding ASCII

Write-Host ""
Write-Host "🎉 Installation Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To use GovDocShield X CLI:" -ForegroundColor Yellow
Write-Host "  1. Activate the environment: .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  2. Run: python govdocshield.py --help" -ForegroundColor White
Write-Host "  3. Or use: .\govdocshield.bat --help" -ForegroundColor White
Write-Host ""
Write-Host "Example commands:" -ForegroundColor Yellow
Write-Host "  python govdocshield.py init" -ForegroundColor White
Write-Host "  python govdocshield.py analyze suspicious_file.pdf" -ForegroundColor White
Write-Host "  python govdocshield.py status" -ForegroundColor White
Write-Host "  python govdocshield.py batch ./files --pattern *.doc" -ForegroundColor White
Write-Host ""
Write-Host "🔬 Quantum-Neuromorphic Defense Ready! 🧠🐝🧬" -ForegroundColor Magenta