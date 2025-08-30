#!/usr/bin/env python3
"""
launch.py - Cross-platform launcher for Multifactor Stock Analysis System
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'launcher_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProjectLauncher:
    def __init__(self):
        self.project_root = Path.cwd()
        self.ui_file = self.project_root / "ui" / "dashboard.html"
        self.backend_script = self.project_root / "src" / "backend" / "app.py"

    def check_environment(self):
        """Check if all required components are installed"""
        logger.info("Checking environment...")

        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            logger.error("Python 3.8+ is required")
            return False

        # Check required packages
        required_packages = [
            'numpy', 'pandas', 'yfinance', 'flask',
            'scikit-learn', 'matplotlib', 'requests'
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            logger.warning(f"Missing packages: {', '.join(missing_packages)}")
            logger.info("Installing missing packages...")
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages)

        return True

    def create_directory_structure(self):
        """Create necessary directories if they don't exist"""
        directories = [
            "data",
            "data/historical",
            "data/scores",
            "data/portfolios",
            "logs",
            "output",
            "output/reports",
            "output/visualizations",
            "config",
            "ui",
            "src",
            "src/regime_detection",
            "src/weight_optimization",
            "src/scoring",
            "src/trend_analysis",
            "src/portfolio",
            "src/backend",
            "tests"
        ]

        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info("Directory structure verified")

    def start_backend(self):
        """Start the Flask/FastAPI backend server"""
        if self.backend_script.exists():
            logger.info("Starting backend server...")
            # Start backend in a subprocess
            backend_process = subprocess.Popen(
                [sys.executable, str(self.backend_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(3)  # Give backend time to start
            return backend_process
        else:
            logger.warning("Backend script not found. Running in UI-only mode.")
            return None

    def launch_ui(self):
        """Open the dashboard in the default web browser"""
        if self.ui_file.exists():
            logger.info(f"Launching UI: {self.ui_file}")
            webbrowser.open(f"file:///{self.ui_file.absolute()}")
        else:
            logger.error(f"UI file not found: {self.ui_file}")
            logger.info("Please save the dashboard HTML as 'ui/dashboard.html'")

    def run(self):
        """Main launch sequence"""
        print("""
        ╔══════════════════════════════════════════╗
        ║   Multifactor Stock Analysis System      ║
        ║         Version 1.0 (70% Complete)       ║
        ╚══════════════════════════════════════════╝
        """)

        # Check environment
        if not self.check_environment():
            logger.error("Environment check failed")
            return

        # Create directories
        self.create_directory_structure()

        # Start backend
        backend_process = self.start_backend()

        # Launch UI
        self.launch_ui()

        # Keep running
        try:
            logger.info("System is running. Press Ctrl+C to stop.")
            if backend_process:
                backend_process.wait()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            if backend_process:
                backend_process.terminate()
            sys.exit(0)


if __name__ == "__main__":
    launcher = ProjectLauncher()
    launcher.run()