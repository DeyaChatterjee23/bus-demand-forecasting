#!/usr/bin/env python3
"""
Entry point script for running the RedBus Demand Forecasting Streamlit app.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run the Streamlit application."""
    # Add src to Python path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))

    # Run Streamlit app
    app_path = Path(__file__).parent / "streamlit_app" / "app.py"

    cmd = [
        "streamlit", "run", str(app_path),
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.fileWatcherType=none"
    ]

    print("🚌 Starting RedBus Demand Forecasting App...")
    print(f"📍 Access the app at: http://localhost:8501")

    subprocess.run(cmd)

if __name__ == "__main__":
    main()
