#!/usr/bin/env python3
"""
GovDocShield X Dashboard Launcher
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the GovDocShield X dashboard"""
    
    # Get the dashboard directory
    dashboard_dir = Path(__file__).parent / "src" / "dashboard"
    app_path = dashboard_dir / "app.py"
    
    if not app_path.exists():
        print("❌ Dashboard app not found at:", app_path)
        sys.exit(1)
    
    print("🛡️ Starting GovDocShield X Dashboard...")
    print("🌐 Dashboard will open in your browser")
    print("💡 Use Ctrl+C to stop the dashboard")
    print()
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false",
            "--theme.base", "dark",
            "--theme.primaryColor", "#667eea",
            "--theme.backgroundColor", "#0e1117",
            "--theme.secondaryBackgroundColor", "#262730"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()