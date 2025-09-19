#!/usr/bin/env python3
"""
GovDocShield X CLI Setup Script
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# Import the main CLI
from cli.main import cli

if __name__ == '__main__':
    cli()